"""Environment configuration is explicit, validated, and safe to display."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from research_rag.errors import ConfigurationError
from research_rag.settings import Settings


def test_environment_paths_resolve_from_the_process_working_directory(
    monkeypatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RAG_DATA_DIR", "runtime-data")
    monkeypatch.setenv("RAG_CORPUS_PATH", "course/papers.yaml")
    monkeypatch.setenv("RAG_GOLDEN_PATH", "course/questions.yaml")

    settings = Settings.from_env()

    assert settings.data_dir == (tmp_path / "runtime-data").resolve()
    assert settings.corpus_path == (tmp_path / "course/papers.yaml").resolve()
    assert (
        settings.golden_questions_path == (tmp_path / "course/questions.yaml").resolve()
    )
    assert settings.index_dir == settings.data_dir / "index"


def test_provider_keys_are_masked_but_available_to_the_clients(monkeypatch):
    secret = "sk-offline-secret-that-must-not-appear"
    qdrant_secret = "qdrant-offline-secret-that-must-not-appear"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    monkeypatch.setenv("QDRANT_API_KEY", qdrant_secret)
    monkeypatch.setenv("QDRANT_URL", "https://qdrant.internal.example")

    settings = Settings.from_env()

    assert secret not in repr(settings)
    assert secret not in str(settings.model_dump())
    assert qdrant_secret not in repr(settings)
    assert qdrant_secret not in str(settings.model_dump())
    assert settings.require_openai_api_key() == secret
    assert settings.qdrant_api_key is not None
    assert settings.qdrant_api_key.get_secret_value() == qdrant_secret


def test_qdrant_backend_environment_is_explicit(monkeypatch):
    monkeypatch.setenv("RAG_VECTOR_BACKEND", "QDRANT")
    monkeypatch.setenv("QDRANT_URL", "https://qdrant.internal.example")
    monkeypatch.setenv("QDRANT_COLLECTION", "course-rag")
    monkeypatch.setenv("QDRANT_TIMEOUT_SECONDS", "17")
    monkeypatch.setenv("QDRANT_INDEX_TIMEOUT_SECONDS", "91")
    monkeypatch.setenv("QDRANT_REPLICATION_FACTOR", "2")
    monkeypatch.setenv("QDRANT_WRITE_CONSISTENCY_FACTOR", "2")
    monkeypatch.setenv("QDRANT_PREFER_GRPC", "on")

    settings = Settings.from_env()

    assert settings.vector_backend == "qdrant"
    assert settings.qdrant_url == "https://qdrant.internal.example"
    assert settings.qdrant_collection == "course-rag"
    assert settings.qdrant_timeout_seconds == 17
    assert settings.qdrant_index_timeout_seconds == 91
    assert settings.qdrant_replication_factor == 2
    assert settings.qdrant_write_consistency_factor == 2
    assert settings.qdrant_prefer_grpc is True


def test_qdrant_write_consistency_cannot_exceed_replication(monkeypatch):
    monkeypatch.setenv("RAG_VECTOR_BACKEND", "qdrant")
    monkeypatch.setenv("QDRANT_REPLICATION_FACTOR", "1")
    monkeypatch.setenv("QDRANT_WRITE_CONSISTENCY_FACTOR", "2")

    with pytest.raises(ConfigurationError, match="cannot exceed"):
        Settings.from_env()


def test_qdrant_api_key_requires_https(monkeypatch):
    monkeypatch.setenv("RAG_VECTOR_BACKEND", "qdrant")
    monkeypatch.setenv("QDRANT_API_KEY", "secret")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.internal.example:6333")

    with pytest.raises(ConfigurationError, match="must use HTTPS"):
        Settings.from_env()


def test_missing_api_key_fails_only_when_a_provider_needs_it(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "   ")

    settings = Settings.from_env()

    assert settings.openai_api_key is None
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY is required"):
        settings.require_openai_api_key()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RAG_QUERY_MODE", "magic"),
        ("RAG_VECTOR_BACKEND", "magic"),
        ("RAG_RERANK", "auto"),
        ("RAG_RERANK", "sometimes"),
        ("RAG_CONTEXT_K", "0"),
        ("QDRANT_TIMEOUT_SECONDS", "0"),
        ("QDRANT_PREFER_GRPC", "auto"),
        ("QDRANT_PREFER_GRPC", "sometimes"),
        ("RAG_API_PORT", "70000"),
        ("OPENAI_TIMEOUT_SECONDS", "-1"),
    ],
)
def test_invalid_environment_values_fail_fast(monkeypatch, name: str, value: str):
    monkeypatch.setenv(name, value)

    with pytest.raises(ConfigurationError):
        Settings.from_env()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        *[
            (name, 0)
            for name in (
                "qdrant_timeout_seconds",
                "qdrant_index_timeout_seconds",
                "qdrant_replication_factor",
                "qdrant_write_consistency_factor",
                "per_query_k",
                "candidate_k",
                "context_k",
                "openai_timeout_seconds",
                "download_timeout_seconds",
                "api_port",
            )
        ],
        *[(name, -1) for name in ("openai_max_retries", "download_retries")],
        *[
            (name, "")
            for name in (
                "embedding_model",
                "caption_model",
                "query_model",
                "rerank_model",
                "generation_model",
                "evaluation_model",
                "api_host",
                "log_level",
            )
        ],
        ("api_port", 65_536),
    ],
)
def test_programmatic_settings_enforce_the_same_field_bounds(
    settings_factory, name: str, value: object
):
    with pytest.raises(ValidationError):
        settings_factory(**{name: value})
