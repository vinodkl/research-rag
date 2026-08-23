"""Advanced retrieval: search several query views, then fuse their rankings.

The vector index still does the broad, cheap search.  Query rewriting and HyDE
only change what we search with; Reciprocal Rank Fusion (RRF) combines the
ranked lists without pretending that cosine scores from different queries are
directly comparable.

The original question is always one of the query views.  Rewrites and a HyDE
document can improve recall, but neither is trusted as evidence: only real
indexed chunks are returned to generation.
"""

from dataclasses import dataclass

from research_rag.models import Chunk
from research_rag.retrieval import store
from research_rag.retrieval.query_planning import QueryVariant

RRF_K = 60
QUERY_WEIGHTS = {"original": 1.0, "rewrite": 0.8, "hyde": 0.7}


@dataclass(frozen=True)
class Candidate:
    """One real chunk found through one or more query variants."""

    chunk: Chunk
    vector_score: float
    guard_score: float
    fusion_score: float
    query_kinds: tuple[str, ...]


def retrieve(
    vector_store: store.VectorStore,
    variants: list[QueryVariant],
    *,
    per_query_k: int = 20,
    candidate_k: int = 20,
    diagnostics: dict[str, str] | None = None,
) -> list[Candidate]:
    """Retrieve for each query variant and fuse the unique chunk rankings.

    ``vector_score`` is the best cosine score seen for a chunk and is kept for
    the evidence floor. ``fusion_score`` is used only for ordering candidates.
    """
    if per_query_k < 1 or candidate_k < 1:
        return []

    if not variants:
        return []

    # Every backend receives the same batch. This is one OpenAI embedding call
    # and, for Qdrant, one server request whether the plan has one or four views.
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

    # chunk id -> mutable aggregate kept local so Candidate can stay frozen.
    combined: dict[str, dict] = {}
    for variant, ranked in zip(variants, ranked_lists, strict=True):
        weight = QUERY_WEIGHTS.get(variant.kind, 0.7)
        for rank, (chunk, vector_score) in enumerate(ranked, start=1):
            item = combined.setdefault(
                chunk.id,
                {
                    "chunk": chunk,
                    "vector_score": vector_score,
                    "guard_score": float("-inf"),
                    "fusion_score": 0.0,
                    "query_kinds": set(),
                },
            )
            item["vector_score"] = max(item["vector_score"], vector_score)
            # A fabricated HyDE document can drift toward a plausible but
            # unrelated neighborhood. It helps ranking, but cannot by itself
            # satisfy the evidence-quality guard.
            if variant.kind != "hyde":
                item["guard_score"] = max(item["guard_score"], vector_score)
            item["fusion_score"] += weight / (RRF_K + rank)
            item["query_kinds"].add(variant.kind)

    candidates = [
        Candidate(
            chunk=item["chunk"],
            vector_score=float(item["vector_score"]),
            guard_score=float(item["guard_score"]),
            fusion_score=float(item["fusion_score"]),
            query_kinds=tuple(sorted(item["query_kinds"])),
        )
        for item in combined.values()
    ]
    candidates.sort(
        key=lambda item: (item.fusion_score, item.vector_score), reverse=True
    )
    return candidates[:candidate_k]


def as_scored_chunks(candidates: list[Candidate]) -> list[tuple[Chunk, float]]:
    """Adapter for reranking/generation while retaining trace data separately."""
    return [(candidate.chunk, candidate.vector_score) for candidate in candidates]
