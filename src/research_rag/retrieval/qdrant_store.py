"""Load and query the server-backed Qdrant vector store.

Serving resolves the stable alias once, validates its collection contract, and
pins that physical build for the process lifetime. Offline collection creation
and alias publication live in :mod:`research_rag.retrieval.qdrant_index`.
"""

import math
import uuid
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
    """A validated physical collection implementing the shared search API."""

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
                # Pin the validated physical build. A rolling restart can pick
                # up a new alias target without changing one under a request.
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
        """Check server and collection state without exposing connection data."""

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


def _check_server_ready(client: QdrantClient) -> None:
    """Use the public SDK surface for both remote and in-memory clients."""

    client.info()


def _aliases(client: QdrantClient) -> dict[str, str]:
    return {
        alias.alias_name: alias.collection_name
        for alias in client.get_aliases().aliases
    }


def _point_id(chunk_id: str) -> uuid.UUID:
    return uuid.uuid5(POINT_NAMESPACE, chunk_id)
