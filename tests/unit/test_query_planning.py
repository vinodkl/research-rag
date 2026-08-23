"""Unit tests use one fake OpenAI Responses call for query planning."""

import json
from types import SimpleNamespace

from research_rag.retrieval.query_planning import (
    DEFAULT_MODEL,
    MAX_VARIANT_CHARS,
    ExpansionOutput,
    QueryVariant,
    expand,
    should_expand,
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


def _output(rewrites=None, hyde_passage="") -> ExpansionOutput:
    return ExpansionOutput(
        rewrites=rewrites or [],
        hyde_passage=hyde_passage,
    )


def test_original_mode_preserves_question_and_makes_no_model_call():
    client = FakeClient(error=AssertionError("must not be called"))

    result = expand("What is retrieval?", "original", client=client)

    assert result.variants == [QueryVariant("What is retrieval?", "original")]
    assert result.strategy_status == {
        "rewrite": "not-requested",
        "hyde": "not-requested",
    }
    assert result.model is None
    assert client.responses.calls == []


def test_rewrite_mode_dedupes_and_honours_limit():
    client = FakeClient(
        _output(
            [
                "What is retrieval?",
                "How does document retrieval work?",
                "Which search method finds relevant passages?",
            ]
        )
    )

    result = expand("What is retrieval?", "rewrite", client=client, max_rewrites=2)

    assert result.variants == [
        QueryVariant("What is retrieval?", "original"),
        QueryVariant("How does document retrieval work?", "rewrite"),
        QueryVariant("Which search method finds relevant passages?", "rewrite"),
    ]
    assert result.strategy_status == {"rewrite": "applied", "hyde": "not-requested"}
    assert len(client.responses.calls) == 1
    request = json.loads(client.responses.calls[0]["input"])
    assert request == {
        "question": "What is retrieval?",
        "create_rewrites": True,
        "maximum_rewrites": 2,
        "create_hyde_passage": False,
    }


def test_hyde_mode_adds_one_retrieval_only_passage():
    passage = "A retriever embeds queries and ranks nearby passages."
    client = FakeClient(_output(hyde_passage=passage))

    result = expand("How does retrieval work?", "hyde", client=client)

    assert result.variants == [
        QueryVariant("How does retrieval work?", "original"),
        QueryVariant(passage, "hyde"),
    ]
    assert result.strategy_status == {"rewrite": "not-requested", "hyde": "applied"}
    assert "never be used as evidence" in client.responses.calls[0]["instructions"]


def test_hybrid_gets_rewrites_and_hyde_from_one_call():
    client = FakeClient(
        _output(
            ["How are retrieved passages selected?"],
            "Dense retrieval ranks passages by vector similarity.",
        )
    )

    result = expand("Explain dense retrieval.", "hybrid", client=client)

    assert [item.kind for item in result.variants] == ["original", "rewrite", "hyde"]
    assert result.strategy_status == {"rewrite": "applied", "hyde": "applied"}
    assert len(client.responses.calls) == 1
    call = client.responses.calls[0]
    assert call["model"] == DEFAULT_MODEL
    assert call["text_format"] is ExpansionOutput
    assert call["reasoning"] == {"effort": "medium"}
    assert call["store"] is False


def test_auto_only_expands_complex_questions():
    simple_client = FakeClient(error=AssertionError("simple question must not expand"))
    assert not should_expand("What is RAG?")
    assert expand("What is RAG?", "auto", client=simple_client).variants == [
        QueryVariant("What is RAG?", "original")
    ]
    assert simple_client.responses.calls == []

    question = "Compare RAG and fine-tuning and explain their trade-offs."
    complex_client = FakeClient(
        _output(
            ["How does RAG differ from fine-tuning?"],
            "RAG retrieves passages, while fine-tuning updates model weights.",
        )
    )
    assert should_expand(question)
    result = expand(question, "auto", client=complex_client)
    assert [item.kind for item in result.variants] == ["original", "rewrite", "hyde"]
    assert len(complex_client.responses.calls) == 1


def test_model_failure_falls_back_to_original_with_safe_diagnostics():
    client = FakeClient(error=RuntimeError("request contained a secret value"))

    result = expand("Explain retrieval and generation.", "hybrid", client=client)

    assert result.variants == [
        QueryVariant("Explain retrieval and generation.", "original")
    ]
    assert result.strategy_status == {
        "rewrite": "unavailable",
        "hyde": "unavailable",
    }
    assert result.errors == ("Query planning failed (RuntimeError)",)
    assert "secret value" not in repr(result)


def test_valid_but_empty_output_records_empty_strategies():
    result = expand(
        "Explain retrieval and generation.",
        "hybrid",
        client=FakeClient(_output()),
    )

    assert result.strategy_status == {"rewrite": "empty", "hyde": "empty"}
    assert [item.kind for item in result.variants] == ["original"]


def test_generated_text_is_bounded_even_if_a_test_double_bypasses_validation():
    output = ExpansionOutput.model_construct(
        rewrites=["x" * (MAX_VARIANT_CHARS + 500)],
        hyde_passage="y" * (MAX_VARIANT_CHARS + 500),
    )

    result = expand(
        "Explain retrieval and generation.", "hybrid", client=FakeClient(output)
    )

    assert [len(item.text) for item in result.variants[1:]] == [
        MAX_VARIANT_CHARS,
        MAX_VARIANT_CHARS,
    ]


def test_schema_bounds_each_rewrite_before_the_response_is_generated():
    schema = ExpansionOutput.model_json_schema()

    assert schema["properties"]["rewrites"]["items"]["maxLength"] == MAX_VARIANT_CHARS


def test_model_can_be_overridden_without_changing_code(monkeypatch):
    monkeypatch.setenv("OPENAI_QUERY_MODEL", "gpt-test-query")
    client = FakeClient(_output(["retrieval query"]))

    result = expand("Explain retrieval.", "rewrite", client=client)

    assert result.model == "gpt-test-query"
    assert client.responses.calls[0]["model"] == "gpt-test-query"


def test_all_supported_modes_keep_original_first():
    for mode in ("original", "rewrite", "hyde", "hybrid", "auto"):
        result = expand(
            "What is RAG?", mode, client=FakeClient(_output()), max_rewrites=0
        )
        assert result.variants[0] == QueryVariant("What is RAG?", "original")


def test_zero_rewrite_limit_skips_a_noop_model_call():
    client = FakeClient(error=AssertionError("no strategy means no model call"))

    result = expand("Explain retrieval.", "rewrite", client=client, max_rewrites=0)

    assert result.variants == [QueryVariant("Explain retrieval.", "original")]
    assert client.responses.calls == []
