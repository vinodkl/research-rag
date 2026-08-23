"""The server-backed Qdrant vector-store implementation.

Each ingestion writes a new physical collection, validates it, and then moves a
stable alias in one atomic operation. The previous collection is retained so an
operator can roll back the alias without re-embedding the corpus.
"""

import hashlib
import json
import math
import time
import uuid
from contextlib import suppress
from dataclasses import asdict
from datetime import UTC, datetime
from functools import lru_cache
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict
from qdrant_client import QdrantClient, models

from research_rag.clients import qdrant_client
from research_rag.errors import IndexUnavailable
from research_rag.models import Chunk
from research_rag.retrieval.embeddings import embed
from research_rag.settings import Settings

SCHEMA_VERSION = 1
POINT_NAMESPACE = uuid.UUID("7736349d-bb11-40b9-bfe1-86bde4626f20")


class QdrantManifest(BaseModel):
    """Embedding and corpus contract stored in collection metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int
    backend: Literal["qdrant"]
    build_id: str
    embedding_model: str
    vector_dimension: int
    chunk_count: int
    corpus_sha256: str
    chunks_sha256: str
    collection_name: str
    alias_name: str
    created_at: str


def build(
    chunks: list[Chunk],
    *,
    corpus_sha256: str | None = None,
    settings: Settings,
    client: QdrantClient | None = None,
) -> QdrantManifest:
    """Build, validate, then atomically publish one Qdrant collection."""

    if not chunks:
        raise ValueError("cannot build an empty index")
    chunk_ids = [chunk.id for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("chunk ids must be unique")

    # Fail before an expensive corpus-wide embedding call when the configured
    # service, credentials, or alias cannot be used.
    active_client = client or qdrant_client(settings)
    _check_server_ready(active_client)
    initial_aliases = _aliases(active_client)
    if (
        settings.qdrant_collection not in initial_aliases
        and active_client.collection_exists(settings.qdrant_collection)
    ):
        raise IndexUnavailable(
            "Qdrant alias name collides with an existing physical collection"
        )

    vectors = embed([chunk.text for chunk in chunks], settings=settings)
    if (
        vectors.ndim != 2
        or vectors.shape[0] != len(chunks)
        or vectors.shape[1] < 1
        or not np.isfinite(vectors).all()
    ):
        raise ValueError("embedding response has an unexpected shape or value")

    chunks_bytes = json.dumps(
        [asdict(chunk) for chunk in chunks], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    chunks_sha256 = hashlib.sha256(chunks_bytes).hexdigest()
    build_id = uuid.uuid4().hex
    physical_name = f"{settings.qdrant_collection}__{build_id}"
    manifest = QdrantManifest(
        schema_version=SCHEMA_VERSION,
        backend="qdrant",
        build_id=build_id,
        embedding_model=settings.embedding_model,
        vector_dimension=int(vectors.shape[1]),
        chunk_count=len(chunks),
        corpus_sha256=corpus_sha256 or chunks_sha256,
        chunks_sha256=chunks_sha256,
        collection_name=physical_name,
        alias_name=settings.qdrant_collection,
        created_at=datetime.now(UTC).isoformat(),
    )

    created = active_client.create_collection(
        collection_name=physical_name,
        vectors_config=models.VectorParams(
            size=manifest.vector_dimension,
            distance=models.Distance.COSINE,
        ),
        metadata=manifest.model_dump(),
        replication_factor=settings.qdrant_replication_factor,
        write_consistency_factor=settings.qdrant_write_consistency_factor,
        timeout=settings.qdrant_timeout_seconds,
    )
    if not created:
        raise IndexUnavailable("Qdrant did not create the new collection")

    # Nothing below this point can affect the currently published alias until
    # the final update_collection_aliases call.
    try:
        active_client.upload_points(
            collection_name=physical_name,
            points=(
                models.PointStruct(
                    id=_point_id(chunk.id),
                    vector=vector.tolist(),
                    payload=asdict(chunk),
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ),
            batch_size=64,
            max_retries=3,
            wait=True,
        )
        _validate_collection(
            active_client,
            physical_name,
            settings,
            expected=manifest,
            exact_count=True,
            require_green=False,
        )
        _validate_sample_payload(active_client, physical_name, chunks[0])
        _wait_until_green(active_client, physical_name, settings)
    except Exception:
        _delete_unpublished(active_client, physical_name, settings)
        raise

    try:
        publish_aliases = _aliases(active_client)
    except Exception:
        _delete_unpublished(active_client, physical_name, settings)
        raise
    if publish_aliases.get(settings.qdrant_collection) != initial_aliases.get(
        settings.qdrant_collection
    ):
        _delete_unpublished(active_client, physical_name, settings)
        raise IndexUnavailable("Qdrant alias changed during ingestion; retry the build")

    operations: list[models.AliasOperations] = []
    if settings.qdrant_collection in publish_aliases:
        operations.append(
            models.DeleteAliasOperation(
                delete_alias=models.DeleteAlias(alias_name=settings.qdrant_collection)
            )
        )
    operations.append(
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(
                collection_name=physical_name,
                alias_name=settings.qdrant_collection,
            )
        )
    )

    # Do not delete the new collection if this call times out: the server may
    # have completed the atomic switch even when the client missed the reply.
    published = active_client.update_collection_aliases(
        change_aliases_operations=operations,
        timeout=settings.qdrant_timeout_seconds,
    )
    if not published:
        raise IndexUnavailable("Qdrant did not publish the new collection alias")
    if _aliases(active_client).get(settings.qdrant_collection) != physical_name:
        raise IndexUnavailable("Qdrant alias publication could not be verified")

    _load_cached.cache_clear()
    return manifest


def load(settings: Settings) -> "QdrantVectorStore":
    """Load and validate the collection behind the configured stable alias."""

    return _load_cached(settings)


def clear_cache() -> None:
    """Release cached store wrappers before their shared clients are closed."""

    _load_cached.cache_clear()


@lru_cache(maxsize=4)
def _load_cached(settings: Settings) -> "QdrantVectorStore":
    active_client = qdrant_client(settings)
    aliases = _aliases(active_client)
    physical_name = aliases.get(settings.qdrant_collection)
    if physical_name is None:
        raise IndexUnavailable("no Qdrant index found; run: research-rag ingest")
    manifest = _validate_collection(
        active_client,
        settings.qdrant_collection,
        settings,
        exact_count=True,
        require_green=True,
    )
    if manifest.collection_name != physical_name:
        raise IndexUnavailable("Qdrant alias and collection manifest do not match")
    return QdrantVectorStore(
        client=active_client,
        settings=settings,
        manifest=manifest,
    )


class QdrantVectorStore:
    """A validated Qdrant alias implementing the shared search contract."""

    backend = "qdrant"

    def __init__(
        self,
        *,
        client: QdrantClient,
        settings: Settings,
        manifest: QdrantManifest,
    ) -> None:
        self.client = client
        self.settings = settings
        self.manifest = manifest

    def search_many(
        self, questions: list[str], *, k: int
    ) -> list[list[tuple[Chunk, float]]]:
        """Use one embedding call and one Qdrant batch-query request."""

        if not questions:
            return []
        vectors = embed(questions, settings=self.settings)
        if (
            vectors.ndim != 2
            or vectors.shape != (len(questions), self.manifest.vector_dimension)
            or not np.isfinite(vectors).all()
        ):
            raise IndexUnavailable("query embeddings do not match the Qdrant index")

        try:
            responses = self.client.query_batch_points(
                # Pin the validated physical build for this process. The alias
                # selects a build at startup; rolling restarts move replicas to
                # a newly published build without changing one under a request.
                collection_name=self.manifest.collection_name,
                requests=[
                    models.QueryRequest(
                        query=vector.tolist(),
                        limit=k,
                        with_payload=True,
                        with_vector=False,
                    )
                    for vector in vectors
                ],
                timeout=self.settings.qdrant_timeout_seconds,
            )
            if len(responses) != len(questions):
                raise ValueError("Qdrant returned the wrong number of rankings")
            return [
                [_scored_chunk(point) for point in response.points]
                for response in responses
            ]
        except IndexUnavailable:
            raise
        except Exception as exc:
            raise IndexUnavailable("Qdrant query failed") from exc

    def readiness(self) -> dict[str, object]:
        """Check server/collection state without exposing connection details."""

        _check_server_ready(self.client)
        info = self.client.get_collection(self.manifest.collection_name)
        manifest = _manifest(info)
        status = str(info.status)
        compatible = (
            status == str(models.CollectionStatus.GREEN)
            and manifest == self.manifest
            and manifest.embedding_model == self.settings.embedding_model
        )
        return {
            "ready": compatible,
            "backend": self.backend,
            "status": status,
            "chunks": manifest.chunk_count,
            "dimension": manifest.vector_dimension,
            "build_id": manifest.build_id,
            "collection": manifest.alias_name,
        }


def _validate_collection(
    client: QdrantClient,
    collection_name: str,
    settings: Settings,
    *,
    expected: QdrantManifest | None = None,
    exact_count: bool,
    require_green: bool,
) -> QdrantManifest:
    try:
        info = client.get_collection(collection_name)
        manifest = _manifest(info)
        vector_config = info.config.params.vectors
        if not isinstance(vector_config, models.VectorParams):
            raise ValueError("named or missing vector configuration")
        if vector_config.distance != models.Distance.COSINE:
            raise ValueError("distance must be cosine")
        if vector_config.size != manifest.vector_dimension:
            raise ValueError("dimension does not match metadata")
        if manifest.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported manifest schema")
        if manifest.embedding_model != settings.embedding_model:
            raise ValueError("embedding model changed")
        if manifest.alias_name != settings.qdrant_collection:
            raise ValueError("alias does not match metadata")
        if require_green and info.status != models.CollectionStatus.GREEN:
            raise ValueError("collection is not ready")
        if expected is not None and manifest != expected:
            raise ValueError("manifest does not match the requested build")
        if exact_count:
            count = client.count(
                collection_name,
                exact=True,
                timeout=settings.qdrant_timeout_seconds,
            ).count
            if count != manifest.chunk_count:
                raise ValueError("point count does not match metadata")
    except IndexUnavailable:
        raise
    except Exception as exc:
        raise IndexUnavailable("Qdrant collection is missing or incompatible") from exc
    return manifest


def _manifest(info: models.CollectionInfo) -> QdrantManifest:
    try:
        return QdrantManifest.model_validate(info.config.metadata)
    except Exception as exc:
        raise IndexUnavailable("Qdrant collection metadata is invalid") from exc


def _scored_chunk(point: models.ScoredPoint) -> tuple[Chunk, float]:
    try:
        if point.payload is None or not math.isfinite(point.score):
            raise ValueError("missing payload or invalid score")
        return Chunk(**point.payload), float(point.score)
    except Exception as exc:
        raise IndexUnavailable("Qdrant returned an invalid chunk payload") from exc


def _validate_sample_payload(
    client: QdrantClient, collection_name: str, expected: Chunk
) -> None:
    try:
        records = client.retrieve(
            collection_name,
            ids=[_point_id(expected.id)],
            with_payload=True,
            with_vectors=False,
        )
        if len(records) != 1 or records[0].payload is None:
            raise ValueError("sample point is missing")
        if Chunk(**records[0].payload) != expected:
            raise ValueError("sample payload changed during storage")
    except Exception as exc:
        raise IndexUnavailable("Qdrant payload validation failed") from exc


def _wait_until_green(
    client: QdrantClient, collection_name: str, settings: Settings
) -> None:
    deadline = time.monotonic() + settings.qdrant_index_timeout_seconds
    while True:
        status = client.get_collection(collection_name).status
        if status == models.CollectionStatus.GREEN:
            return
        if status == models.CollectionStatus.RED:
            raise IndexUnavailable("Qdrant collection entered a failed state")
        if time.monotonic() >= deadline:
            raise IndexUnavailable("Qdrant collection did not become ready in time")
        time.sleep(0.25)


def _check_server_ready(client: QdrantClient) -> None:
    try:
        client.http.service_api.readyz()
    except NotImplementedError:
        # In-memory Qdrant is permitted only in offline tests and has no REST API.
        return


def _aliases(client: QdrantClient) -> dict[str, str]:
    return {
        alias.alias_name: alias.collection_name
        for alias in client.get_aliases().aliases
    }


def _point_id(chunk_id: str) -> uuid.UUID:
    return uuid.uuid5(POINT_NAMESPACE, chunk_id)


def _delete_unpublished(
    client: QdrantClient, collection_name: str, settings: Settings
) -> None:
    with suppress(Exception):
        client.delete_collection(
            collection_name, timeout=settings.qdrant_timeout_seconds
        )
