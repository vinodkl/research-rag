"""Vector-store boundary: one interface, two explicit backends.

The rest of the RAG pipeline knows only how to search a batch of questions. The
factory below chooses local FAISS or server-backed Qdrant from configuration;
it never silently falls back to a different backend when one is unavailable.
"""

from typing import Protocol

from research_rag.models import Chunk
from research_rag.settings import Settings, get_settings

ScoredChunk = tuple[Chunk, float]


class VectorStore(Protocol):
    """The complete online contract required by retrieval."""

    backend: str

    def search_many(
        self, questions: list[str], *, k: int
    ) -> list[list[ScoredChunk]]: ...

    def readiness(self) -> dict[str, object]: ...


class BuildResult(Protocol):
    """The one build field ingestion displays, independent of backend."""

    build_id: str


def build(
    chunks: list[Chunk],
    *,
    corpus_sha256: str | None = None,
    settings: Settings | None = None,
) -> BuildResult:
    """Build only the configured backend; backend outages are never hidden."""

    active = settings or get_settings()
    if active.vector_backend == "faiss":
        from research_rag.retrieval import faiss_store

        return faiss_store.build(chunks, corpus_sha256=corpus_sha256, settings=active)

    from research_rag.retrieval import qdrant_index

    return qdrant_index.build(chunks, corpus_sha256=corpus_sha256, settings=active)


def load(settings: Settings | None = None) -> VectorStore:
    """Load the configured backend and validate its embedding contract."""

    active = settings or get_settings()
    if active.vector_backend == "faiss":
        from research_rag.retrieval import faiss_store

        return faiss_store.load(active)

    from research_rag.retrieval import qdrant_store

    return qdrant_store.load(active)


def readiness(settings: Settings | None = None) -> dict[str, object]:
    """Return safe backend status for health endpoints and evaluation reports."""

    active = settings or get_settings()
    try:
        return load(active).readiness()
    except Exception as exc:  # paths, URLs, and provider messages stay private
        return {
            "ready": False,
            "backend": active.vector_backend,
            "error": type(exc).__name__,
        }


def clear_cache() -> None:
    """Clear concrete backend caches during application shutdown."""

    from research_rag.retrieval import faiss_store, qdrant_store

    faiss_store.clear_cache()
    qdrant_store.clear_cache()
