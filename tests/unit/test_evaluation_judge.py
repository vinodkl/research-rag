"""Unit tests for the structured RAG judge and Python-side aggregation."""

import json
from types import SimpleNamespace

import pytest

from research_rag.evaluation.judge import evaluate
from research_rag.ingestion.chunking import Chunk


def _chunk(chunk_id: str, text: str = "Evidence") -> Chunk:
    return Chunk(
        id=chunk_id,
        paper="paper",
        title="A Paper",
        section="2 Method",
        page=2,
        text=text,
    )


class FakeCompletions:
    def __init__(self, payload=None, *, content=None, error=None):
        self.payload = payload
        self.content = content
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        content = self.content if self.content is not None else json.dumps(self.payload)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class FakeClient:
    def __init__(self, payload=None, *, content=None, error=None):
        self.completions = FakeCompletions(payload, content=content, error=error)
        self.chat = SimpleNamespace(completions=self.completions)


def test_parses_one_call_and_computes_aggregates_in_python():
    client = FakeClient(
        {
            "context_scores": [
                {"chunk_id": "p:1", "score": 0.8, "reason": "Direct evidence."},
                {"chunk_id": "p:2", "score": 0.4, "reason": "Only background."},
            ],
            "answer_relevance": {"score": 0.75, "reason": "Direct but incomplete."},
            "claims": [
                {"claim": "Claim one", "supported": True},
                {"claim": "Claim two", "supported": False},
                {"claim": "Claim three", "supported": True},
            ],
        }
    )

    result = evaluate(
        "What happened?", "Three claims.", [_chunk("p:1"), _chunk("p:2")], client=client
    )

    assert len(client.completions.calls) == 1
    assert client.completions.calls[0]["response_format"]["type"] == "json_schema"
    assert result.context_relevance.score == pytest.approx(0.6)
    assert result.answer_relevance.score == 0.75
    assert result.faithfulness.score == pytest.approx(2 / 3)
    assert result.supported_claims == 2
    assert result.total_claims == 3
    assert result.unsupported_claims == ["Claim two"]


def test_clamps_scores_and_ignores_duplicate_unknown_and_malformed_entries():
    client = FakeClient(
        {
            "context_scores": [
                {"chunk_id": "p:1", "score": 2.5, "reason": "Very useful."},
                {"chunk_id": "p:1", "score": 0.1, "reason": "Duplicate."},
                {"chunk_id": "p:2", "score": -3, "reason": "Not useful."},
                {"chunk_id": "ghost", "score": 1, "reason": "Unknown."},
                {"chunk_id": "p:3", "score": "high", "reason": "Malformed."},
                "not an object",
            ],
            "answer_relevance": {"score": -2, "reason": "Off topic."},
            "claims": [
                {"claim": "Supported claim", "supported": True},
                {"claim": " supported   claim ", "supported": False},
                {"claim": "Unsupported claim", "supported": False},
                {"claim": "Bad flag", "supported": 1},
            ],
        }
    )

    result = evaluate(
        "Question?",
        "An answer.",
        [_chunk("p:1"), _chunk("p:2"), _chunk("p:3")],
        client=client,
    )

    assert [item.score for item in result.contexts] == [1.0, 0.0, 0.0]
    assert result.context_relevance.score == pytest.approx(1 / 3)
    assert result.answer_relevance.score == 0.0
    assert result.faithfulness.score == 0.5
    assert result.total_claims == 2
    assert result.unsupported_claims == ["Unsupported claim"]


def test_missing_context_scores_are_zero_and_input_context_ids_are_unique():
    client = FakeClient(
        {
            "context_scores": [],
            "answer_relevance": {"score": 1, "reason": "Relevant."},
            "claims": [{"claim": "One claim", "supported": True}],
        }
    )

    result = evaluate(
        "Question?",
        "One claim.",
        [_chunk("p:1"), _chunk("p:1"), _chunk("p:2")],
        client=client,
    )

    assert [item.chunk_id for item in result.contexts] == ["p:1", "p:2"]
    assert [item.score for item in result.contexts] == [0.0, 0.0]
    assert result.context_relevance.score == 0.0


@pytest.mark.parametrize(
    "answer",
    [
        "",
        "   ",
    ],
)
def test_empty_answer_is_scored_locally_without_a_provider_call(answer):
    client = FakeClient(error=AssertionError("provider must not be called"))

    result = evaluate("Question?", answer, [_chunk("p:1")], client=client)

    assert client.completions.calls == []
    assert result.answer_relevance.score == 0.0
    assert result.faithfulness.score is None
    assert result.context_relevance.score == 0.0
    assert result.contexts[0].score == 0.0
    assert result.total_claims == 0


def test_refused_answer_still_judges_context_relevance():
    client = FakeClient(
        {
            "context_scores": [
                {"chunk_id": "p:1", "score": 0.8, "reason": "Useful evidence."}
            ],
            # These answer fields are required by the schema but ignored for a refusal.
            "answer_relevance": {"score": 0.9, "reason": "Ignore this."},
            "claims": [{"claim": "Ignore this too.", "supported": True}],
        }
    )

    result = evaluate(
        "Question?",
        "I could not produce an answer with verifiable citations.",
        [_chunk("p:1")],
        client=client,
    )

    assert len(client.completions.calls) == 1
    assert result.context_relevance.score == 0.8
    assert result.contexts[0].score == 0.8
    assert result.answer_relevance.score == 0.0
    assert result.faithfulness.score is None
    assert result.total_claims == 0


def test_refused_answer_without_context_is_local_and_context_is_undefined():
    client = FakeClient(error=AssertionError("provider must not be called"))

    result = evaluate(
        "Question?", "I could not find enough evidence.", [], client=client
    )

    assert client.completions.calls == []
    assert result.context_relevance.score is None
    assert result.answer_relevance.score == 0.0
    assert result.faithfulness.score is None


def test_explicit_refusal_flag_overrides_wording_heuristics():
    payload = {
        "context_scores": [
            {"chunk_id": "p:1", "score": 0.7, "reason": "Useful evidence."}
        ],
        "answer_relevance": {"score": 0.6, "reason": "Partially direct."},
        "claims": [{"claim": "A supported claim.", "supported": True}],
    }
    non_refusal_client = FakeClient(payload)
    non_refusal = evaluate(
        "Question?",
        "I cannot establish the exact value, but the paper reports an improvement.",
        [_chunk("p:1")],
        client=non_refusal_client,
        refused=False,
    )
    assert non_refusal.answer_relevance.score == 0.6
    assert non_refusal.faithfulness.score == 1.0

    refusal_client = FakeClient(payload)
    refusal = evaluate(
        "Question?",
        "There is insufficient evidence in the indexed papers to answer.",
        [_chunk("p:1")],
        client=refusal_client,
        refused=True,
    )
    assert refusal.context_relevance.score == 0.7
    assert refusal.answer_relevance.score == 0.0
    assert refusal.faithfulness.score is None


def test_no_contexts_has_an_undefined_context_metric():
    client = FakeClient(
        {
            "context_scores": [],
            "answer_relevance": {"score": 0.9, "reason": "Direct."},
            "claims": [{"claim": "An unsupported claim", "supported": False}],
        }
    )

    result = evaluate("Question?", "An unsupported claim.", [], client=client)

    assert result.context_relevance.score is None
    assert result.contexts == []
    assert result.answer_relevance.score == 0.9
    assert result.faithfulness.score == 0.0


def test_provider_failure_returns_unavailable_metrics_without_raising():
    client = FakeClient(error=RuntimeError("secret provider detail"))

    result = evaluate("Question?", "Answer.", [_chunk("p:1")], client=client)

    assert result.context_relevance.score is None
    assert result.answer_relevance.score is None
    assert result.faithfulness.score is None
    assert result.contexts == []
    assert "secret provider detail" not in result.context_relevance.reason


def test_invalid_json_returns_unavailable_metrics_without_raising():
    result = evaluate(
        "Question?",
        "Answer.",
        [_chunk("p:1")],
        client=FakeClient(content="not json"),
    )

    assert result.context_relevance.score is None
    assert result.answer_relevance.score is None
    assert result.faithfulness.score is None
    assert "invalid response" in result.faithfulness.reason
