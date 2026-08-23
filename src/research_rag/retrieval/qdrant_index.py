"""Build and atomically publish versioned Qdrant collections.

This is the offline half of the Qdrant backend. Ingestion creates a physical
collection, uploads and validates every chunk, waits for it to become ready,
and only then moves the stable application alias. The previous collection is
retained for rollback.
"""

import hashlib
import json
import time
import uuid
from contextlib import suppress
from dataclasses import asdict
from datetime import UTC, datetime

import numpy as np
from qdrant_client import QdrantClient, models

from research_rag.clients import qdrant_client
from research_rag.errors import IndexUnavailable
from research_rag.models import Chunk
from research_rag.retrieval.embeddings import embed
from research_rag.retrieval.qdrant_store import (
    SCHEMA_VERSION,
    QdrantManifest,
    _aliases,
    _check_server_ready,
    _point_id,
    _validate_collection,
    clear_cache,
)
from research_rag.settings import Settings


def build(
    chunks: list[Chunk],
    *,
    corpus_sha256: str | None = None,
    settings: Settings,
    client: QdrantClient | None = None,
) -> QdrantManifest:
    """Build, validate, then atomically publish one Qdrant collection."""

    _validate_chunks(chunks)
    active_client = client or qdrant_client(settings)
    initial_aliases = _preflight(active_client, settings)
    vectors = _embed_chunks(chunks, settings)
    manifest = _new_manifest(chunks, vectors, corpus_sha256, settings)

    _create_collection(active_client, manifest, settings)
    _upload_and_validate(active_client, chunks, vectors, manifest, settings)
    _publish_alias(active_client, manifest, initial_aliases, settings)

    # A new load must resolve the alias again instead of returning the build
    # that this process validated before ingestion.
    clear_cache()
    return manifest


def _validate_chunks(chunks: list[Chunk]) -> None:
    if not chunks:
        raise ValueError("cannot build an empty index")
    chunk_ids = [chunk.id for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("chunk ids must be unique")


def _preflight(client: QdrantClient, settings: Settings) -> dict[str, str]:
    """Check connectivity and remember the alias target before embedding."""

    _check_server_ready(client)
    aliases = _aliases(client)
    alias = settings.qdrant_collection
    if alias not in aliases and client.collection_exists(alias):
        raise IndexUnavailable(
            "Qdrant alias name collides with an existing physical collection"
        )
    return aliases


def _embed_chunks(chunks: list[Chunk], settings: Settings) -> np.ndarray:
    vectors = embed([chunk.text for chunk in chunks], settings=settings)
    if (
        vectors.ndim != 2
        or vectors.shape[0] != len(chunks)
        or vectors.shape[1] < 1
        or not np.isfinite(vectors).all()
    ):
        raise ValueError("embedding response has an unexpected shape or value")
    return vectors


def _new_manifest(
    chunks: list[Chunk],
    vectors: np.ndarray,
    corpus_sha256: str | None,
    settings: Settings,
) -> QdrantManifest:
    chunks_bytes = json.dumps(
        [asdict(chunk) for chunk in chunks], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    chunks_sha256 = hashlib.sha256(chunks_bytes).hexdigest()
    build_id = uuid.uuid4().hex
    physical_name = f"{settings.qdrant_collection}__{build_id}"
    return QdrantManifest(
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


def _create_collection(
    client: QdrantClient, manifest: QdrantManifest, settings: Settings
) -> None:
    created = client.create_collection(
        collection_name=manifest.collection_name,
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


def _upload_and_validate(
    client: QdrantClient,
    chunks: list[Chunk],
    vectors: np.ndarray,
    manifest: QdrantManifest,
    settings: Settings,
) -> None:
    """Prepare a complete build without changing the published alias."""

    try:
        client.upload_points(
            collection_name=manifest.collection_name,
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
            client,
            manifest.collection_name,
            settings,
            expected=manifest,
            exact_count=True,
            require_green=False,
        )
        _validate_sample_payload(client, manifest.collection_name, chunks[0])
        _wait_until_green(client, manifest.collection_name, settings)
    except Exception:
        _delete_unpublished(client, manifest.collection_name, settings)
        raise


def _publish_alias(
    client: QdrantClient,
    manifest: QdrantManifest,
    initial_aliases: dict[str, str],
    settings: Settings,
) -> None:
    """Publish only if no other ingestion moved the alias meanwhile."""

    alias = settings.qdrant_collection
    try:
        current_aliases = _aliases(client)
    except Exception:
        _delete_unpublished(client, manifest.collection_name, settings)
        raise

    if current_aliases.get(alias) != initial_aliases.get(alias):
        _delete_unpublished(client, manifest.collection_name, settings)
        raise IndexUnavailable("Qdrant alias changed during ingestion; retry the build")

    operations: list[models.AliasOperations] = []
    if alias in current_aliases:
        operations.append(
            models.DeleteAliasOperation(
                delete_alias=models.DeleteAlias(alias_name=alias)
            )
        )
    operations.append(
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(
                collection_name=manifest.collection_name,
                alias_name=alias,
            )
        )
    )

    # Do not delete the new collection after this call starts: the server may
    # have completed the atomic switch even if the client misses the response.
    published = client.update_collection_aliases(
        change_aliases_operations=operations,
        timeout=settings.qdrant_timeout_seconds,
    )
    if not published:
        raise IndexUnavailable("Qdrant did not publish the new collection alias")
    if _aliases(client).get(alias) != manifest.collection_name:
        raise IndexUnavailable("Qdrant alias publication could not be verified")


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


def _delete_unpublished(
    client: QdrantClient, collection_name: str, settings: Settings
) -> None:
    with suppress(Exception):
        client.delete_collection(
            collection_name, timeout=settings.qdrant_timeout_seconds
        )
