"""Search query views and combine their rankings with weighted RRF.

Rewrites and HyDE can improve recall, but only indexed chunks are returned.
HyDE similarity is tracked separately so synthetic text cannot prove relevance.
"""

from dataclasses import dataclass

from research_rag.models import Chunk
from research_rag.retrieval import store
from research_rag.retrieval.query_planning import QueryVariant

RRF_K = 60
QUERY_WEIGHTS = {"original": 1.0, "rewrite": 0.8, "hyde": 0.7}


@dataclass(frozen=True)
class Candidate:
    """One real chunk found through one or more query variants.

    Best similarity may include HyDE; real-query score excludes it. RRF score
    orders candidates without comparing cosine scores across query views.
    """

    chunk: Chunk
    best_similarity: float
    real_query_score: float
    rrf_score: float
    query_kinds: tuple[str, ...]


def retrieve(
    vector_store: store.VectorStore,
    variants: list[QueryVariant],
    *,
    per_query_k: int = 20,
    candidate_k: int = 20,
    diagnostics: dict[str, str] | None = None,
) -> list[Candidate]:
    """Search every query view, retry the original on failure, then fuse."""
    if per_query_k < 1 or candidate_k < 1 or not variants:
        return []

    try:
        ranked_lists = vector_store.search_many(
            [variant.text for variant in variants], k=per_query_k
        )
        if len(ranked_lists) != len(variants):
            raise ValueError("batched retrieval returned the wrong number of rankings")
    except Exception as exc:  # retry the trusted original query
        original = next(
            (variant for variant in variants if variant.kind == "original"), None
        )
        if original is None or len(variants) == 1:
            raise
        if diagnostics is not None:
            diagnostics["batch_error"] = (
                f"Expanded retrieval failed ({type(exc).__name__}); original query retried"
            )
        variants = [original]
        ranked_lists = vector_store.search_many([original.text], k=per_query_k)
        if len(ranked_lists) != 1:
            raise ValueError(
                "original-query retry returned an invalid ranking"
            ) from exc

    combined: dict[str, dict] = {}
    for variant, ranked in zip(variants, ranked_lists, strict=True):
        weight = QUERY_WEIGHTS.get(variant.kind, 0.7)
        for rank, (chunk, similarity) in enumerate(ranked, start=1):
            item = combined.setdefault(
                chunk.id,
                {
                    "chunk": chunk,
                    "best_similarity": similarity,
                    "real_query_score": float("-inf"),
                    "rrf_score": 0.0,
                    "query_kinds": set(),
                },
            )
            item["best_similarity"] = max(item["best_similarity"], similarity)
            if variant.kind != "hyde":
                item["real_query_score"] = max(item["real_query_score"], similarity)
            item["rrf_score"] += weight / (RRF_K + rank)
            item["query_kinds"].add(variant.kind)

    candidates = [
        Candidate(
            chunk=item["chunk"],
            best_similarity=float(item["best_similarity"]),
            real_query_score=float(item["real_query_score"]),
            rrf_score=float(item["rrf_score"]),
            query_kinds=tuple(sorted(item["query_kinds"])),
        )
        for item in combined.values()
    ]
    candidates.sort(
        key=lambda item: (item.rrf_score, item.best_similarity), reverse=True
    )
    return candidates[:candidate_k]


def as_scored_chunks(candidates: list[Candidate]) -> list[tuple[Chunk, float]]:
    """Adapter for reranking/generation while retaining trace data separately."""
    return [(candidate.chunk, candidate.best_similarity) for candidate in candidates]
