"""Unit tests for OpenAI listwise reranking use a fake Responses client."""

import json
from types import SimpleNamespace

import pytest

from research_rag.ingestion.chunking import Chunk
from research_rag.retrieval import reranker as rerank_module
from research_rag.retrieval.reranker import (
    DEFAULT_MODEL,
    RankedChunk,
    RankingOutput,
    rerank,
)


def _chunk(identifier: str, text: str) -> Chunk:
    return Chunk(
        id=identifier,
        paper="tiny",
        title="Tiny Paper",
        section="2 Method",
        page=2,
        text=text,
    )


class FakeResponses:
    def __init__(self, output=None, *, error=None):
        self.output = output
        self.error = error
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(output_parsed=self.output)


class FakeClient:
    def __init__(self, output=None, *, error=None):
        self.responses = FakeResponses(output, error=error)


def _ranking(*items: tuple[str, str]) -> RankingOutput:
    return RankingOutput(
        ranking=[
            RankedChunk(chunk_id=chunk_id, relevance=relevance)
            for chunk_id, relevance in items
        ]
    )


def test_one_listwise_call_maps_order_and_coarse_grades():
    first = _chunk("tiny:0", "first passage")
    second = _chunk("tiny:1", "second passage")
    third = _chunk("tiny:2", "third passage")
    client = FakeClient(
        _ranking(
            (third.id, "direct"),
            (first.id, "supporting"),
            (second.id, "irrelevant"),
        )
    )

    result = rerank(
        "Which passage is best?",
        [(first, 0.8), (second, 0.7), (third, 0.6)],
        top_k=2,
        client=client,
    )

    assert result.applied
    assert result.provider == "openai"
    assert result.model == DEFAULT_MODEL
    assert result.error is None
    assert result.results == [(third, 1.0), (first, 0.5)]
    assert len(client.responses.calls) == 1
    call = client.responses.calls[0]
    request = json.loads(call["input"])
    assert request["question"] == "Which passage is best?"
    assert [item["chunk_id"] for item in request["candidates"]] == [
        first.id,
        second.id,
        third.id,
    ]
    assert [item["text"] for item in request["candidates"]] == [
        first.text,
        second.text,
        third.text,
    ]
    assert call["text_format"] is RankingOutput
    assert call["reasoning"] == {"effort": "medium"}
    assert call["store"] is False
    assert "untrusted data" in call["instructions"]


def test_duplicate_input_chunk_ids_are_sent_once():
    first = _chunk("tiny:0", "keep the first copy")
    duplicate = _chunk("tiny:0", "drop the duplicate")
    second = _chunk("tiny:1", "another chunk")
    client = FakeClient(_ranking((first.id, "direct"), (second.id, "supporting")))

    result = rerank(
        "question",
        [(first, 0.8), (duplicate, 0.7), (second, 0.6)],
        client=client,
    )

    request = json.loads(client.responses.calls[0]["input"])
    assert [item["text"] for item in request["candidates"]] == [
        "keep the first copy",
        "another chunk",
    ]
    assert result.results == [(first, 1.0), (second, 0.5)]


@pytest.mark.parametrize(
    "output",
    [
        _ranking(("tiny:0", "direct")),  # missing an ID
        _ranking(("tiny:0", "direct"), ("tiny:0", "supporting")),
        _ranking(("tiny:0", "direct"), ("ghost", "supporting")),
    ],
)
def test_malformed_candidate_permutation_falls_back_as_a_whole(output):
    first = _chunk("tiny:0", "first")
    second = _chunk("tiny:1", "second")

    result = rerank(
        "question", [(first, 0.8), (second, 0.7)], client=FakeClient(output)
    )

    assert not result.applied
    assert result.results == [(first, 0.8), (second, 0.7)]
    assert result.error == "OpenAI reranking failed (ValueError)"


def test_all_irrelevant_is_an_applied_empty_ranking_not_a_fallback():
    first = _chunk("tiny:0", "first")
    second = _chunk("tiny:1", "second")

    result = rerank(
        "question",
        [(first, 0.8), (second, 0.7)],
        client=FakeClient(
            _ranking((first.id, "irrelevant"), (second.id, "irrelevant"))
        ),
    )

    assert result.applied
    assert result.results == []
    assert result.error is None


def test_direct_grade_outranks_contradictory_model_order():
    supporting = _chunk("tiny:supporting", "background")
    direct = _chunk("tiny:direct", "answer-bearing evidence")
    client = FakeClient(_ranking((supporting.id, "supporting"), (direct.id, "direct")))

    result = rerank(
        "question", [(supporting, 0.8), (direct, 0.7)], top_k=1, client=client
    )

    assert result.applied
    assert result.results == [(direct, 1.0)]


def test_none_parsed_output_falls_back_cleanly():
    first = _chunk("tiny:0", "first")

    result = rerank("question", [(first, 0.8)], client=FakeClient(None))

    assert not result.applied
    assert result.results == [(first, 0.8)]
    assert result.error == "OpenAI reranking failed (ValueError)"


def test_request_failure_falls_back_without_leaking_details():
    first = _chunk("tiny:0", "first")
    second = _chunk("tiny:1", "second")
    client = FakeClient(error=RuntimeError("request contained super-secret-value"))

    result = rerank("question", [(first, 0.8), (second, 0.7)], top_k=1, client=client)

    assert not result.applied
    assert result.results == [(first, 0.8)]
    assert result.error == "OpenAI reranking failed (RuntimeError)"
    assert "super-secret-value" not in result.error


def test_missing_openai_key_falls_back(monkeypatch):
    first = _chunk("tiny:0", "first")
    monkeypatch.setattr(
        rerank_module,
        "openai_client",
        lambda *_: (_ for _ in ()).throw(RuntimeError("OPENAI_API_KEY is not set")),
    )

    result = rerank("question", [(first, 0.8)])

    assert not result.applied
    assert result.results == [(first, 0.8)]
    assert result.error == "OpenAI reranking failed (RuntimeError)"


@pytest.mark.parametrize(("candidates", "top_k"), [([], 6), ([(None, 0.0)], 0)])
def test_empty_or_zero_limit_makes_no_model_call(candidates, top_k):
    client = FakeClient(error=AssertionError("must not call OpenAI"))
    if candidates:
        candidates = [(_chunk("tiny:0", "first"), 0.8)]

    result = rerank("question", candidates, top_k=top_k, client=client)

    assert result.results == []
    assert not result.applied
    assert result.error is None
    assert client.responses.calls == []


def test_model_can_be_overridden_without_changing_code(monkeypatch):
    first = _chunk("tiny:0", "first")
    monkeypatch.setenv("OPENAI_RERANK_MODEL", "gpt-test-reranker")
    client = FakeClient(_ranking((first.id, "direct")))

    result = rerank("question", [(first, 0.8)], client=client)

    assert result.model == "gpt-test-reranker"
    assert client.responses.calls[0]["model"] == "gpt-test-reranker"
