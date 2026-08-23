"""All environment-driven configuration in one inspectable place.

The defaults make a source checkout easy to teach from. A deployed process can
set explicit paths and model names without changing application code.
"""

import os
from pathlib import Path
from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from research_rag.errors import ConfigurationError

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_CAPTION_MODEL = "gpt-5.6-luna"
DEFAULT_QUERY_MODEL = "gpt-5.6-sol"
DEFAULT_RERANK_MODEL = "gpt-5.6-sol"
DEFAULT_GENERATION_MODEL = "gpt-5.6-sol"
DEFAULT_EVALUATION_MODEL = "gpt-5.6-luna"

QueryMode = Literal["auto", "original", "rewrite", "hyde", "hybrid"]
VectorBackend = Literal["faiss", "qdrant"]


class Settings(BaseModel):
    """Validated process settings. Secrets stay masked in logs and reprs."""

    model_config = ConfigDict(frozen=True)

    openai_api_key: SecretStr | None
    data_dir: Path
    corpus_path: Path
    golden_questions_path: Path

    # FAISS keeps the lesson runnable with no infrastructure. Qdrant provides
    # the server-backed option used by a multi-replica deployment.
    vector_backend: VectorBackend = "faiss"
    qdrant_url: str = Field(default="http://127.0.0.1:6333", pattern=r"^https?://")
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = Field(
        default="research-rag",
        min_length=1,
        max_length=120,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    qdrant_timeout_seconds: int = Field(default=10, gt=0)
    qdrant_index_timeout_seconds: int = Field(default=300, gt=0)
    qdrant_replication_factor: int = Field(default=1, ge=1)
    qdrant_write_consistency_factor: int = Field(default=1, ge=1)
    qdrant_prefer_grpc: bool = False

    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL, min_length=1)
    caption_model: str = Field(default=DEFAULT_CAPTION_MODEL, min_length=1)
    query_model: str = Field(default=DEFAULT_QUERY_MODEL, min_length=1)
    rerank_model: str = Field(default=DEFAULT_RERANK_MODEL, min_length=1)
    generation_model: str = Field(default=DEFAULT_GENERATION_MODEL, min_length=1)
    evaluation_model: str = Field(default=DEFAULT_EVALUATION_MODEL, min_length=1)

    query_mode: QueryMode = "auto"
    rerank_enabled: bool = True
    per_query_k: int = Field(default=20, gt=0)
    candidate_k: int = Field(default=20, gt=0)
    context_k: int = Field(default=6, gt=0)

    openai_timeout_seconds: float = Field(default=60.0, gt=0)
    openai_max_retries: int = Field(default=2, ge=0)
    download_timeout_seconds: float = Field(default=60.0, gt=0)
    download_retries: int = Field(default=2, ge=0)
    api_host: str = Field(default="127.0.0.1", min_length=1)
    api_port: int = Field(default=8477, ge=1, le=65_535)
    log_level: str = Field(default="INFO", min_length=1)

    @model_validator(mode="after")
    def validate_qdrant_security_and_consistency(self) -> Self:
        """Apply the same safety rules to env and programmatic settings."""

        if self.vector_backend != "qdrant":
            return self
        if self.qdrant_api_key is not None and not self.qdrant_url.startswith(
            "https://"
        ):
            raise ValueError("QDRANT_URL must use HTTPS when QDRANT_API_KEY is set")
        if self.qdrant_write_consistency_factor > self.qdrant_replication_factor:
            raise ValueError(
                "QDRANT_WRITE_CONSISTENCY_FACTOR cannot exceed replication factor"
            )
        return self

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def pdf_dir(self) -> Path:
        return self.data_dir / "pdfs"

    @property
    def caption_cache_path(self) -> Path:
        return self.data_dir / "captions.json"

    def require_openai_api_key(self) -> str:
        if self.openai_api_key is None:
            raise ConfigurationError(
                "OPENAI_API_KEY is required; copy .env.example and export it"
            )
        return self.openai_api_key.get_secret_value()

    @classmethod
    def from_env(cls) -> "Settings":
        """Read and validate the current process environment."""

        query_mode = _text("RAG_QUERY_MODE", "auto").lower()
        if query_mode not in {"auto", "original", "rewrite", "hyde", "hybrid"}:
            raise ConfigurationError(
                "RAG_QUERY_MODE must be auto, original, rewrite, hyde, or hybrid"
            )

        vector_backend = _text("RAG_VECTOR_BACKEND", "faiss").lower()
        if vector_backend not in {"faiss", "qdrant"}:
            raise ConfigurationError("RAG_VECTOR_BACKEND must be faiss or qdrant")

        qdrant_url = _text("QDRANT_URL", "http://127.0.0.1:6333")
        if not qdrant_url.startswith(("http://", "https://")):
            raise ConfigurationError("QDRANT_URL must begin with http:// or https://")
        qdrant_replication_factor = _positive_int("QDRANT_REPLICATION_FACTOR", 1)
        qdrant_write_consistency_factor = _positive_int(
            "QDRANT_WRITE_CONSISTENCY_FACTOR", 1
        )
        if (
            vector_backend == "qdrant"
            and qdrant_write_consistency_factor > qdrant_replication_factor
        ):
            raise ConfigurationError(
                "QDRANT_WRITE_CONSISTENCY_FACTOR cannot exceed replication factor"
            )

        key = os.getenv("OPENAI_API_KEY", "").strip()
        qdrant_key = os.getenv("QDRANT_API_KEY", "").strip()
        if (
            vector_backend == "qdrant"
            and qdrant_key
            and not qdrant_url.startswith("https://")
        ):
            raise ConfigurationError(
                "QDRANT_URL must use HTTPS when QDRANT_API_KEY is set"
            )
        return cls(
            openai_api_key=SecretStr(key) if key else None,
            data_dir=_path("RAG_DATA_DIR", _checkout_or_cwd("data")),
            corpus_path=_path(
                "RAG_CORPUS_PATH", _checkout_or_cwd("config/papers.yaml")
            ),
            golden_questions_path=_path(
                "RAG_GOLDEN_PATH",
                _checkout_or_cwd("config/golden_questions.yaml"),
            ),
            vector_backend=cast(VectorBackend, vector_backend),
            qdrant_url=qdrant_url,
            qdrant_api_key=SecretStr(qdrant_key) if qdrant_key else None,
            qdrant_collection=_text("QDRANT_COLLECTION", "research-rag"),
            qdrant_timeout_seconds=_positive_int("QDRANT_TIMEOUT_SECONDS", 10),
            qdrant_index_timeout_seconds=_positive_int(
                "QDRANT_INDEX_TIMEOUT_SECONDS", 300
            ),
            qdrant_replication_factor=qdrant_replication_factor,
            qdrant_write_consistency_factor=qdrant_write_consistency_factor,
            qdrant_prefer_grpc=_boolean("QDRANT_PREFER_GRPC", False),
            embedding_model=_text("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            caption_model=_text("OPENAI_CAPTION_MODEL", DEFAULT_CAPTION_MODEL),
            query_model=_text("OPENAI_QUERY_MODEL", DEFAULT_QUERY_MODEL),
            rerank_model=_text("OPENAI_RERANK_MODEL", DEFAULT_RERANK_MODEL),
            generation_model=_text("OPENAI_GENERATION_MODEL", DEFAULT_GENERATION_MODEL),
            evaluation_model=_text("OPENAI_EVAL_MODEL", DEFAULT_EVALUATION_MODEL),
            query_mode=cast(QueryMode, query_mode),
            rerank_enabled=_boolean("RAG_RERANK", True),
            per_query_k=_positive_int("RAG_PER_QUERY_K", 20),
            candidate_k=_positive_int("RAG_CANDIDATE_K", 20),
            context_k=_positive_int("RAG_CONTEXT_K", 6),
            openai_timeout_seconds=_positive_float("OPENAI_TIMEOUT_SECONDS", 60.0),
            openai_max_retries=_non_negative_int("OPENAI_MAX_RETRIES", 2),
            download_timeout_seconds=_positive_float(
                "RAG_DOWNLOAD_TIMEOUT_SECONDS", 60.0
            ),
            download_retries=_non_negative_int("RAG_DOWNLOAD_RETRIES", 2),
            api_host=_text("RAG_API_HOST", "127.0.0.1"),
            api_port=_port("RAG_API_PORT", 8477),
            log_level=_text("RAG_LOG_LEVEL", "INFO").upper(),
        )


def get_settings() -> Settings:
    """Return fresh settings so tests and CLI overrides remain unsurprising."""

    return Settings.from_env()


def _checkout_or_cwd(relative: str) -> Path:
    checkout = Path(__file__).resolve().parents[2]
    root = checkout if (checkout / "pyproject.toml").is_file() else Path.cwd()
    return (root / relative).resolve()


def _path(name: str, default: Path) -> Path:
    value = os.getenv(name, "").strip()
    return Path(value).expanduser().resolve() if value else default


def _text(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def _boolean(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be on or off")


def _positive_int(name: str, default: int) -> int:
    return _bounded_int(name, default, minimum=1)


def _non_negative_int(name: str, default: int) -> int:
    return _bounded_int(name, default, minimum=0)


def _port(name: str, default: int) -> int:
    return _bounded_int(name, default, minimum=1, maximum=65_535)


def _bounded_int(
    name: str, default: int, *, minimum: int, maximum: int | None = None
) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if value < minimum or (maximum is not None and value > maximum):
        raise ConfigurationError(f"{name} is outside its allowed range")
    return value


def _positive_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number") from exc
    if value <= 0:
        raise ConfigurationError(f"{name} must be greater than zero")
    return value
