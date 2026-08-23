"""The configured backend is explicit and failures never trigger fallback."""

import pytest

from research_rag.retrieval import faiss_store, qdrant_index, qdrant_store, store


def test_load_dispatches_only_to_the_configured_qdrant_backend(
    monkeypatch, settings_factory
):
    settings = settings_factory(vector_backend="qdrant")
    expected = object()

    def unexpected(*_args, **_kwargs):
        raise AssertionError("Qdrant failure must not fall back to FAISS")

    monkeypatch.setattr(faiss_store, "load", unexpected)
    monkeypatch.setattr(qdrant_store, "load", lambda received: expected)

    assert store.load(settings) is expected


def test_build_dispatches_only_to_the_configured_faiss_backend(
    monkeypatch, settings_factory
):
    settings = settings_factory(vector_backend="faiss")
    chunks = []
    expected = object()
    calls = []

    def fake_build(received, *, corpus_sha256, settings):
        calls.append((received, corpus_sha256, settings))
        return expected

    def unexpected(*_args, **_kwargs):
        raise AssertionError("FAISS selection must not call Qdrant")

    monkeypatch.setattr(faiss_store, "build", fake_build)
    monkeypatch.setattr(qdrant_index, "build", unexpected)

    assert store.build(chunks, corpus_sha256="a" * 64, settings=settings) is expected
    assert calls == [(chunks, "a" * 64, settings)]


def test_readiness_redacts_backend_exception_details(monkeypatch, settings_factory):
    settings = settings_factory(vector_backend="qdrant")

    def unavailable(_settings):
        raise RuntimeError("https://secret.internal:6333?api_key=private")

    monkeypatch.setattr(store, "load", unavailable)

    status = store.readiness(settings)

    assert status == {
        "ready": False,
        "backend": "qdrant",
        "error": "RuntimeError",
    }
    assert "secret.internal" not in repr(status)
    assert "private" not in repr(status)


def test_qdrant_load_failure_is_not_hidden_by_faiss_fallback(
    monkeypatch, settings_factory
):
    settings = settings_factory(vector_backend="qdrant")

    def qdrant_failure(_settings):
        raise RuntimeError("Qdrant is unavailable")

    def unexpected(*_args, **_kwargs):
        raise AssertionError("FAISS must not hide a configured Qdrant outage")

    monkeypatch.setattr(qdrant_store, "load", qdrant_failure)
    monkeypatch.setattr(faiss_store, "load", unexpected)

    with pytest.raises(RuntimeError, match="Qdrant is unavailable"):
        store.load(settings)
