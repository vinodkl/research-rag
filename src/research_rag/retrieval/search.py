"""Advanced retrieval: search several query views, then fuse their rankings.

The vector index still does the broad, cheap search.  Query rewriting and HyDE
only change what we search with; Reciprocal Rank Fusion (RRF) combines the
ranked lists without pretending that cosine scores from different queries are
directly comparable.

The original question is always one of the query views.  Rewrites and a HyDE
document can improve recall, but neither is trusted as evidence: only real
indexed chunks are returned to generation.
"""

from dataclasses import dataclass, field

from research_rag.models import Chunk
from research_rag.retrieval import store
from research_rag.retrieval.query_planning import QueryVariant

RRF_K = 60
QUERY_WEIGHTS = {"original": 1.0, "rewrite": 0.8, "hyde": 0.7}


@dataclass(frozen=True)
class Candidate:
    """One real chunk found through one or more query variants.

    ``best_similarity`` may come from any query, including HyDE.
    ``real_query_score`` excludes HyDE and is safe for the evidence guard.
    ``rrf_score`` combines ranks and is used only to order candidates.
    """

    chunk: Chunk
    best_similarity: float
    real_query_score: float
    rrf_score: float
    query_kinds: tuple[str, ...]


@dataclass
class _CandidateAccumulator:
    """Mutable scores while several ranked lists are being fused."""

    chunk: Chunk
    best_similarity: float = float("-inf")
    real_query_score: float = float("-inf")
    rrf_score: float = 0.0
    query_kinds: set[str] = field(default_factory=set)

    def add(self, *, kind: str, rank: int, similarity: float) -> None:
        self.best_similarity = max(self.best_similarity, similarity)
        if kind != "hyde":
            self.real_query_score = max(self.real_query_score, similarity)
        self.rrf_score += QUERY_WEIGHTS.get(kind, 0.7) / (RRF_K + rank)
        self.query_kinds.add(kind)

    def freeze(self) -> Candidate:
        return Candidate(
            chunk=self.chunk,
            best_similarity=float(self.best_similarity),
            real_query_score=float(self.real_query_score),
            rrf_score=float(self.rrf_score),
            query_kinds=tuple(sorted(self.query_kinds)),
        )


def retrieve(
    vector_store: store.VectorStore,
    variants: list[QueryVariant],
    *,
    per_query_k: int = 20,
    candidate_k: int = 20,
    diagnostics: dict[str, str] | None = None,
) -> list[Candidate]:
    """Retrieve for each query variant and fuse the unique chunk rankings.

    ``best_similarity`` is the best cosine score from any query view.
    ``real_query_score`` excludes HyDE and is safe for the evidence floor.
    ``rrf_score`` is used only for ordering candidates.
    """
    if per_query_k < 1 or candidate_k < 1:
        return []

    if not variants:
        return []

    searched_variants, ranked_lists = _search_variants(
        vector_store,
        variants,
        per_query_k=per_query_k,
        diagnostics=diagnostics,
    )
    return _fuse_rankings(searched_variants, ranked_lists)[:candidate_k]


def _search_variants(
    vector_store: store.VectorStore,
    variants: list[QueryVariant],
    *,
    per_query_k: int,
    diagnostics: dict[str, str] | None,
) -> tuple[list[QueryVariant], list[list[tuple[Chunk, float]]]]:
    """Search one batch; retry only the trusted original if expansion fails."""

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
    return variants, ranked_lists


def _fuse_rankings(
    variants: list[QueryVariant],
    ranked_lists: list[list[tuple[Chunk, float]]],
) -> list[Candidate]:
    """Fuse ranks without comparing cosine scores across different queries."""

    combined: dict[str, _CandidateAccumulator] = {}
    for variant, ranked in zip(variants, ranked_lists, strict=True):
        for rank, (chunk, similarity) in enumerate(ranked, start=1):
            accumulator = combined.setdefault(
                chunk.id, _CandidateAccumulator(chunk=chunk)
            )
            accumulator.add(kind=variant.kind, rank=rank, similarity=similarity)

    candidates = [item.freeze() for item in combined.values()]
    candidates.sort(
        key=lambda item: (item.rrf_score, item.best_similarity), reverse=True
    )
    return candidates


def as_scored_chunks(candidates: list[Candidate]) -> list[tuple[Chunk, float]]:
    """Adapter for reranking/generation while retaining trace data separately."""
    return [(candidate.chunk, candidate.best_similarity) for candidate in candidates]
