"""Offline contract tests for the server-backed Qdrant vector store."""

import numpy as np
import pytest
from qdrant_client import QdrantClient

from research_rag.errors import IndexUnavailable
from research_rag.ingestion.chunking import Chunk
from research_rag.retrieval import qdrant_index, qdrant_store


def _chunk(identifier: str, text: str, *, page: int = 1) -> Chunk:
    return Chunk(
        id=identifier,
        paper="paper",
        title="A Test Paper",
        section="2 Method",
        page=page,
        text=text,
    )


def _vector(text: str) -> list[float]:
    normalized = text.lower()
    if "alpha" in normalized:
        return [1.0, 0.0, 0.0]
    if "beta" in normalized:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


@pytest.fixture
def qdrant_client() -> QdrantClient:
    client = QdrantClient(":memory:")
    yield client
    client.close()


@pytest.fixture(autouse=True)
def clear_qdrant_load_cache():
    qdrant_store._load_cached.cache_clear()
    yield
    qdrant_store._load_cached.cache_clear()


def test_build_load_search_and_readiness_are_backend_neutral(
    monkeypatch, settings_factory, qdrant_client
):
    settings = settings_factory(
        vector_backend="qdrant",
        embedding_model="offline-embedding-v1",
        qdrant_collection="research-rag-test",
    )
    chunks = [
        _chunk("paper:alpha", "alpha evidence"),
        _chunk("paper:beta", "beta evidence", page=2),
    ]
    embedding_calls: list[list[str]] = []

    def fake_embeddings(texts: list[str], **_kwargs) -> np.ndarray:
        embedding_calls.append(texts)
        return np.asarray([_vector(text) for text in texts], dtype="float32")

    monkeypatch.setattr(qdrant_index, "embed", fake_embeddings)
    monkeypatch.setattr(qdrant_store, "embed", fake_embeddings)
    monkeypatch.setattr(qdrant_store, "qdrant_client", lambda _settings: qdrant_client)

    manifest = qdrant_index.build(
        chunks,
        corpus_sha256="a" * 64,
        settings=settings,
        client=qdrant_client,
    )
    loaded = qdrant_store.load(settings)
    rankings = loaded.search_many(["alpha question", "beta question"], k=1)

    assert manifest.backend == "qdrant"
    assert manifest.embedding_model == "offline-embedding-v1"
    assert manifest.vector_dimension == 3
    assert manifest.chunk_count == 2
    assert manifest.corpus_sha256 == "a" * 64
    assert manifest.alias_name == "research-rag-test"
    assert embedding_calls == [
        ["alpha evidence", "beta evidence"],
        ["alpha question", "beta question"],
    ]
    assert [ranking[0][0] for ranking in rankings] == chunks
    assert all(ranking[0][1] == pytest.approx(1.0) for ranking in rankings)
    assert loaded.readiness() == {
        "ready": True,
        "backend": "qdrant",
        "status": "green",
        "chunks": 2,
        "dimension": 3,
        "build_id": manifest.build_id,
        "collection": "research-rag-test",
    }


def test_duplicate_chunk_ids_are_rejected_before_embedding(
    monkeypatch, settings_factory, qdrant_client
):
    settings = settings_factory(vector_backend="qdrant")
    duplicate = _chunk("paper:duplicate", "same id")

    def unexpected(*_args, **_kwargs):
        raise AssertionError("duplicate ids must fail before model usage")

    monkeypatch.setattr(qdrant_index, "embed", unexpected)

    with pytest.raises(ValueError, match="chunk ids must be unique"):
        qdrant_index.build(
            [duplicate, _chunk("paper:duplicate", "different text")],
            settings=settings,
            client=qdrant_client,
        )


def test_qdrant_preflight_fails_before_expensive_embedding(
    monkeypatch, settings_factory, qdrant_client
):
    settings = settings_factory(vector_backend="qdrant")

    def unexpected(*_args, **_kwargs):
        raise AssertionError("embedding must not run after a failed preflight")

    monkeypatch.setattr(qdrant_index, "embed", unexpected)
    monkeypatch.setattr(
        qdrant_index,
        "_check_server_ready",
        lambda _client: (_ for _ in ()).throw(ConnectionError("offline")),
    )

    with pytest.raises(ConnectionError, match="offline"):
        qdrant_index.build(
            [_chunk("paper:alpha", "alpha evidence")],
            settings=settings,
            client=qdrant_client,
        )


def test_load_rejects_an_index_built_with_another_embedding_model(
    monkeypatch, settings_factory, qdrant_client
):
    settings = settings_factory(
        vector_backend="qdrant",
        embedding_model="offline-embedding-v1",
    )
    monkeypatch.setattr(
        qdrant_index,
        "embed",
        lambda texts, **_kwargs: np.asarray(
            [_vector(text) for text in texts], dtype="float32"
        ),
    )
    monkeypatch.setattr(qdrant_store, "qdrant_client", lambda _settings: qdrant_client)
    qdrant_index.build(
        [_chunk("paper:alpha", "alpha evidence")],
        settings=settings,
        client=qdrant_client,
    )
    incompatible = settings.model_copy(
        update={"embedding_model": "offline-embedding-v2"}
    )

    with pytest.raises(IndexUnavailable, match="missing or incompatible"):
        qdrant_store.load(incompatible)


def test_alias_switch_is_atomic_and_keeps_the_previous_collection_for_rollback(
    monkeypatch, settings_factory, qdrant_client
):
    settings = settings_factory(
        vector_backend="qdrant",
        embedding_model="offline-embedding-v1",
    )
    monkeypatch.setattr(
        qdrant_index,
        "embed",
        lambda texts, **_kwargs: np.asarray(
            [_vector(text) for text in texts], dtype="float32"
        ),
    )

    first = qdrant_index.build(
        [_chunk("paper:alpha", "alpha evidence")],
        settings=settings,
        client=qdrant_client,
    )
    second = qdrant_index.build(
        [_chunk("paper:beta", "beta evidence")],
        settings=settings,
        client=qdrant_client,
    )
    aliases = {
        alias.alias_name: alias.collection_name
        for alias in qdrant_client.get_aliases().aliases
    }

    assert aliases[settings.qdrant_collection] == second.collection_name
    assert first.collection_name != second.collection_name
    assert qdrant_client.collection_exists(first.collection_name)
    assert qdrant_client.collection_exists(second.collection_name)
    assert qdrant_client.count(first.collection_name, exact=True).count == 1
    assert qdrant_client.count(second.collection_name, exact=True).count == 1


def test_loaded_process_pins_its_validated_build_until_restart(
    monkeypatch, settings_factory, qdrant_client
):
    settings = settings_factory(
        vector_backend="qdrant",
        embedding_model="offline-embedding-v1",
    )
    monkeypatch.setattr(
        qdrant_index,
        "embed",
        lambda texts, **_kwargs: np.asarray(
            [_vector(text) for text in texts], dtype="float32"
        ),
    )
    monkeypatch.setattr(qdrant_store, "qdrant_client", lambda _settings: qdrant_client)
    alpha = _chunk("paper:alpha", "alpha evidence")
    beta = _chunk("paper:beta", "beta evidence")

    monkeypatch.setattr(
        qdrant_store,
        "embed",
        lambda texts, **_kwargs: np.asarray(
            [_vector(text) for text in texts], dtype="float32"
        ),
    )
    qdrant_index.build([alpha], settings=settings, client=qdrant_client)
    old_process = qdrant_store.load(settings)
    qdrant_index.build([beta], settings=settings, client=qdrant_client)
    restarted_process = qdrant_store.load(settings)

    assert old_process.search_many(["alpha question"], k=1)[0][0][0] == alpha
    assert restarted_process.search_many(["beta question"], k=1)[0][0][0] == beta
    assert old_process.manifest.build_id != restarted_process.manifest.build_id


def test_failed_alias_publication_preserves_the_previously_published_alias(
    monkeypatch, settings_factory, qdrant_client
):
    settings = settings_factory(
        vector_backend="qdrant",
        embedding_model="offline-embedding-v1",
    )
    monkeypatch.setattr(
        qdrant_index,
        "embed",
        lambda texts, **_kwargs: np.asarray(
            [_vector(text) for text in texts], dtype="float32"
        ),
    )
    first = qdrant_index.build(
        [_chunk("paper:alpha", "alpha evidence")],
        settings=settings,
        client=qdrant_client,
    )
    monkeypatch.setattr(
        qdrant_client,
        "update_collection_aliases",
        lambda **_kwargs: False,
    )

    with pytest.raises(IndexUnavailable, match="did not publish"):
        qdrant_index.build(
            [_chunk("paper:beta", "beta evidence")],
            settings=settings,
            client=qdrant_client,
        )

    aliases = {
        alias.alias_name: alias.collection_name
        for alias in qdrant_client.get_aliases().aliases
    }
    assert aliases[settings.qdrant_collection] == first.collection_name
    assert qdrant_client.collection_exists(first.collection_name)


def test_concurrent_alias_change_aborts_and_removes_the_unpublished_build(
    monkeypatch, settings_factory, qdrant_client
):
    settings = settings_factory(
        vector_backend="qdrant",
        embedding_model="offline-embedding-v1",
    )
    monkeypatch.setattr(
        qdrant_index,
        "embed",
        lambda texts, **_kwargs: np.asarray(
            [_vector(text) for text in texts], dtype="float32"
        ),
    )
    first = qdrant_index.build(
        [_chunk("paper:alpha", "alpha evidence")],
        settings=settings,
        client=qdrant_client,
    )
    real_aliases = qdrant_index._aliases
    calls = 0

    def changed_aliases(client):
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_aliases(client)
        return {settings.qdrant_collection: "another-job-build"}

    monkeypatch.setattr(qdrant_index, "_aliases", changed_aliases)

    with pytest.raises(IndexUnavailable, match="changed during ingestion"):
        qdrant_index.build(
            [_chunk("paper:beta", "beta evidence")],
            settings=settings,
            client=qdrant_client,
        )

    aliases = {
        alias.alias_name: alias.collection_name
        for alias in qdrant_client.get_aliases().aliases
    }
    collections = {
        collection.name for collection in qdrant_client.get_collections().collections
    }
    assert aliases[settings.qdrant_collection] == first.collection_name
    assert collections == {first.collection_name}


def test_invalid_point_payload_fails_closed(
    monkeypatch, settings_factory, qdrant_client
):
    settings = settings_factory(
        vector_backend="qdrant",
        embedding_model="offline-embedding-v1",
    )
    chunk = _chunk("paper:alpha", "alpha evidence")
    monkeypatch.setattr(
        qdrant_index,
        "embed",
        lambda texts, **_kwargs: np.asarray(
            [_vector(text) for text in texts], dtype="float32"
        ),
    )
    monkeypatch.setattr(
        qdrant_store,
        "embed",
        lambda texts, **_kwargs: np.asarray(
            [_vector(text) for text in texts], dtype="float32"
        ),
    )
    manifest = qdrant_index.build([chunk], settings=settings, client=qdrant_client)
    qdrant_client.overwrite_payload(
        collection_name=manifest.collection_name,
        payload={"id": ["not-a-string"]},
        points=[qdrant_store._point_id(chunk.id)],
        wait=True,
    )
    loaded = qdrant_store.QdrantVectorStore(
        client=qdrant_client,
        settings=settings,
        manifest=manifest,
    )

    with pytest.raises(IndexUnavailable, match="invalid chunk payload"):
        loaded.search_many(["alpha question"], k=1)
