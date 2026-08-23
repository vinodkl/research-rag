"""The persisted FAISS index is versioned, validated, and atomically selected."""

import json
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np
import pytest

from research_rag.errors import IndexUnavailable
from research_rag.ingestion.chunking import Chunk
from research_rag.retrieval import faiss_store as store


def _chunk(identifier: str, text: str) -> Chunk:
    return Chunk(
        id=identifier,
        paper="paper",
        title="A Test Paper",
        section="2 Method",
        page=3,
        text=text,
    )


def _fake_embeddings(texts: list[str], **_kwargs) -> np.ndarray:
    """Deterministic unit vectors with no API client or key."""

    rows = []
    for position, _text in enumerate(texts):
        rows.append([1.0, 0.0] if position % 2 == 0 else [0.0, 1.0])
    return np.asarray(rows, dtype="float32")


def test_versioned_manifest_round_trip(monkeypatch, settings_factory):
    settings = settings_factory(embedding_model="offline-embedding-v1")
    chunks = [_chunk("paper:0", "alpha"), _chunk("paper:1", "beta")]
    monkeypatch.setattr(store, "embed", _fake_embeddings)

    manifest = store.build(chunks, corpus_sha256="a" * 64, settings=settings)

    build_dir = settings.index_dir / "builds" / manifest.build_id
    assert (settings.index_dir / "CURRENT").read_text().strip() == manifest.build_id
    assert {item.name for item in build_dir.iterdir()} == {
        "chunks.json",
        "index.faiss",
        "manifest.json",
    }
    recorded = json.loads((build_dir / "manifest.json").read_text())
    assert recorded["schema_version"] == store.SCHEMA_VERSION
    assert recorded["embedding_model"] == "offline-embedding-v1"
    assert recorded["vector_dimension"] == 2
    assert recorded["chunk_count"] == 2
    assert recorded["corpus_sha256"] == "a" * 64

    restored = store.load(settings)
    assert restored.backend == "faiss"
    assert restored.index.ntotal == 2
    assert restored.index.d == 2
    assert restored.chunks == chunks
    assert restored.readiness() == {
        "ready": True,
        "backend": "faiss",
        "chunks": 2,
        "dimension": 2,
        "build_id": manifest.build_id,
    }


def test_manifest_checksum_detects_corrupt_chunk_metadata(
    monkeypatch, settings_factory
):
    settings = settings_factory()
    monkeypatch.setattr(store, "embed", _fake_embeddings)
    manifest = store.build([_chunk("paper:0", "alpha")], settings=settings)
    chunks_path = settings.index_dir / "builds" / manifest.build_id / "chunks.json"
    chunks_path.write_text("[]", encoding="utf-8")
    store._load_cached.cache_clear()

    with pytest.raises(IndexUnavailable, match="checksum"):
        store.load(settings)


def test_manifest_rejects_a_different_embedding_model(monkeypatch, settings_factory):
    settings = settings_factory(embedding_model="offline-embedding-v1")
    monkeypatch.setattr(store, "embed", _fake_embeddings)
    store.build([_chunk("paper:0", "alpha")], settings=settings)
    incompatible = settings.model_copy(
        update={"embedding_model": "offline-embedding-v2"}
    )
    store._load_cached.cache_clear()

    with pytest.raises(IndexUnavailable, match="embedding model changed"):
        store.load(incompatible)


def test_legacy_two_file_index_remains_loadable(settings_factory):
    settings = settings_factory()
    settings.index_dir.mkdir(parents=True)
    chunks = [_chunk("legacy:0", "old but valid")]
    index = faiss.IndexFlatIP(2)
    index.add(np.asarray([[1.0, 0.0]], dtype="float32"))
    faiss.write_index(index, str(settings.index_dir / "index.faiss"))
    (settings.index_dir / "chunks.json").write_text(
        json.dumps([asdict(chunk) for chunk in chunks]), encoding="utf-8"
    )
    store._load_cached.cache_clear()

    restored = store.load(settings)

    assert restored.index.ntotal == 1
    assert restored.chunks == chunks


def test_failed_pointer_publication_preserves_the_previous_build(
    monkeypatch, settings_factory
):
    settings = settings_factory()
    monkeypatch.setattr(store, "embed", _fake_embeddings)
    original_chunks = [_chunk("paper:0", "published")]
    first = store.build(original_chunks, settings=settings)

    def fail_pointer_write(_path: Path, _content: str) -> None:
        raise OSError("simulated pointer write failure")

    monkeypatch.setattr(store, "_atomic_write_text", fail_pointer_write)
    with pytest.raises(OSError, match="pointer write failure"):
        store.build(
            [_chunk("paper:1", "new"), _chunk("paper:2", "not published")],
            settings=settings,
        )

    assert (settings.index_dir / "CURRENT").read_text().strip() == first.build_id
    assert not list((settings.index_dir / "builds").glob(".staging-*"))
    store._load_cached.cache_clear()
    restored = store.load(settings)
    assert restored.index.ntotal == 1
    assert restored.chunks == original_chunks
