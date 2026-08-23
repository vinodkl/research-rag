"""Unit tests for backend-neutral retrieval and Reciprocal Rank Fusion."""

import math
from collections.abc import Callable

import pytest

from research_rag.ingestion.chunking import Chunk
from research_rag.retrieval import search as retrieval
from research_rag.retrieval.query_planning import QueryVariant


def _chunk(id: str, text: str | None = None) -> Chunk:
    return Chunk(
        id=id,
        paper="tiny",
        title="Tiny Paper",
        section="2 Method",
        page=2,
        text=text or f"passage {id}",
    )


class FakeVectorStore:
    """Small test double for the one method retrieval is allowed to use."""

    backend = "fake"

    def __init__(
        self,
        results: list[list[tuple[Chunk, float]]]
        | Callable[[list[str], int], list[list[tuple[Chunk, float]]]],
    ) -> None:
        self.results = results
        self.calls: list[tuple[list[str], int]] = []

    def search_many(
        self, questions: list[str], *, k: int
    ) -> list[list[tuple[Chunk, float]]]:
        self.calls.append((questions, k))
        if callable(self.results):
            return self.results(questions, k)
        return self.results

    def readiness(self) -> dict[str, object]:
        return {"ready": True, "backend": self.backend}


def test_one_query_uses_the_backend_neutral_batch_contract():
    chunks = [_chunk("tiny:0"), _chunk("tiny:1")]
    vector_store = FakeVectorStore([[(chunks[1], 0.83), (chunks[0], 0.72)]])

    results = retrieval.retrieve(
        vector_store,
        [QueryVariant("original question", "original")],
        per_query_k=7,
        candidate_k=5,
    )

    assert vector_store.calls == [(["original question"], 7)]
    assert [item.chunk.id for item in results] == ["tiny:1", "tiny:0"]
    assert results[0].best_similarity == pytest.approx(0.83)
    assert results[0].real_query_score == pytest.approx(0.83)
    assert results[0].rrf_score == pytest.approx(1.0 / (retrieval.RRF_K + 1))
    assert results[0].query_kinds == ("original",)


def test_multiple_queries_use_one_batch_and_apply_query_kind_weights():
    original = _chunk("tiny:original")
    rewritten = _chunk("tiny:rewrite")
    hypothetical = _chunk("tiny:hyde")
    variants = [
        QueryVariant("original question", "original"),
        QueryVariant("rewritten question", "rewrite"),
        QueryVariant("hypothetical answer", "hyde"),
    ]
    vector_store = FakeVectorStore(
        [[(original, 0.5)], [(rewritten, 0.5)], [(hypothetical, 0.5)]]
    )

    results = retrieval.retrieve(vector_store, variants, per_query_k=4, candidate_k=3)

    assert vector_store.calls == [
        (
            ["original question", "rewritten question", "hypothetical answer"],
            4,
        )
    ]
    assert [item.chunk.id for item in results] == [
        "tiny:original",
        "tiny:rewrite",
        "tiny:hyde",
    ]
    denominator = retrieval.RRF_K + 1
    assert [item.rrf_score for item in results] == pytest.approx(
        [
            retrieval.QUERY_WEIGHTS["original"] / denominator,
            retrieval.QUERY_WEIGHTS["rewrite"] / denominator,
            retrieval.QUERY_WEIGHTS["hyde"] / denominator,
        ]
    )


def test_rrf_deduplicates_orders_and_records_provenance():
    first = _chunk("tiny:a")
    second = _chunk("tiny:b")
    third = _chunk("tiny:c")
    variants = [
        QueryVariant("original", "original"),
        QueryVariant("rewrite", "rewrite"),
    ]
    vector_store = FakeVectorStore(
        [
            [(first, 0.80), (second, 0.70)],
            [(second, 0.95), (first, 0.60), (third, 0.50)],
        ]
    )

    results = retrieval.retrieve(vector_store, variants)

    assert [item.chunk.id for item in results] == ["tiny:a", "tiny:b", "tiny:c"]
    assert len({item.chunk.id for item in results}) == 3
    by_id = {item.chunk.id: item for item in results}
    assert by_id["tiny:a"].rrf_score == pytest.approx(
        1.0 / (retrieval.RRF_K + 1) + 0.8 / (retrieval.RRF_K + 2)
    )
    assert by_id["tiny:b"].rrf_score == pytest.approx(
        1.0 / (retrieval.RRF_K + 2) + 0.8 / (retrieval.RRF_K + 1)
    )
    assert by_id["tiny:a"].query_kinds == ("original", "rewrite")
    assert by_id["tiny:b"].query_kinds == ("original", "rewrite")
    assert by_id["tiny:c"].query_kinds == ("rewrite",)
    assert by_id["tiny:b"].best_similarity == pytest.approx(0.95)


def test_candidate_cap_is_applied_after_fusion():
    chunks = [_chunk(f"tiny:{number}") for number in range(4)]
    vector_store = FakeVectorStore(
        [[(chunk, 0.9 - number / 10) for number, chunk in enumerate(chunks)]]
    )

    results = retrieval.retrieve(
        vector_store,
        [QueryVariant("question", "original")],
        per_query_k=10,
        candidate_k=2,
    )

    assert [item.chunk.id for item in results] == ["tiny:0", "tiny:1"]


def test_best_similarity_is_separate_from_real_query_score():
    shared = _chunk("tiny:shared")
    hyde_only = _chunk("tiny:hyde-only")
    variants = [
        QueryVariant("original", "original"),
        QueryVariant("rewrite", "rewrite"),
        QueryVariant("hypothetical", "hyde"),
    ]
    vector_store = FakeVectorStore(
        [
            [(shared, 0.41)],
            [(shared, 0.62)],
            [(shared, 0.99), (hyde_only, 0.98)],
        ]
    )

    results = retrieval.retrieve(vector_store, variants)
    by_id = {item.chunk.id: item for item in results}

    assert by_id["tiny:shared"].best_similarity == pytest.approx(0.99)
    assert by_id["tiny:shared"].real_query_score == pytest.approx(0.62)
    assert by_id["tiny:shared"].query_kinds == ("hyde", "original", "rewrite")
    assert by_id["tiny:hyde-only"].best_similarity == pytest.approx(0.98)
    assert math.isinf(by_id["tiny:hyde-only"].real_query_score)
    assert by_id["tiny:hyde-only"].real_query_score < 0
    scored_shared = next(
        score
        for chunk, score in retrieval.as_scored_chunks(results)
        if chunk.id == shared.id
    )
    assert scored_shared == pytest.approx(0.99)


@pytest.mark.parametrize(
    ("per_query_k", "candidate_k"),
    [(0, 1), (-1, 1), (1, 0), (1, -1)],
)
def test_invalid_limits_return_empty_without_searching(per_query_k, candidate_k):
    vector_store = FakeVectorStore([])

    assert (
        retrieval.retrieve(
            vector_store,
            [QueryVariant("question", "original")],
            per_query_k=per_query_k,
            candidate_k=candidate_k,
        )
        == []
    )
    assert vector_store.calls == []


def test_empty_variants_and_empty_rankings_return_empty():
    vector_store = FakeVectorStore([])
    assert retrieval.retrieve(vector_store, []) == []
    assert vector_store.calls == []

    vector_store = FakeVectorStore([[]])
    assert (
        retrieval.retrieve(vector_store, [QueryVariant("question", "original")]) == []
    )
    assert vector_store.calls == [(["question"], 20)]


def test_failed_expanded_batch_retries_only_the_original_query():
    evidence = _chunk("tiny:original")
    diagnostics = {}
    variants = [
        QueryVariant("trusted original", "original"),
        QueryVariant("malformed expansion", "rewrite"),
    ]

    def results(questions: list[str], _k: int):
        if len(questions) > 1:
            raise RuntimeError("secret request detail")
        return [[(evidence, 0.7)]]

    vector_store = FakeVectorStore(results)
    retrieved = retrieval.retrieve(
        vector_store,
        variants,
        per_query_k=9,
        diagnostics=diagnostics,
    )

    assert vector_store.calls == [
        (["trusted original", "malformed expansion"], 9),
        (["trusted original"], 9),
    ]
    assert [item.chunk.id for item in retrieved] == [evidence.id]
    assert retrieved[0].query_kinds == ("original",)
    assert diagnostics == {
        "batch_error": "Expanded retrieval failed (RuntimeError); original query retried"
    }
    assert "secret request detail" not in repr(diagnostics)


def test_backend_returning_wrong_number_of_rankings_is_rejected():
    vector_store = FakeVectorStore([])

    with pytest.raises(ValueError, match="wrong number"):
        retrieval.retrieve(
            vector_store,
            [QueryVariant("question", "original")],
        )
