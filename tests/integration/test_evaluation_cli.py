"""Integration tests for the end-to-end evaluation command."""

import json
import sys
from collections import Counter

from research_rag.evaluation import runner as eval_cli
from research_rag.evaluation.judge import Evaluation, Metric
from research_rag.ingestion.chunking import Chunk
from research_rag.pipeline import Answer, PipelineTrace


def _answer(
    question: str,
    paper: str,
    *,
    citation_paper: str | None = None,
) -> Answer:
    chunk = Chunk(
        id=f"{paper}:0",
        paper=paper,
        title=f"{paper} paper",
        section="1 Method",
        page=1,
        text=f"Evidence from {paper}.",
    )
    trace = PipelineTrace(
        sanitized_question=question,
        redactions=(),
        query_mode="hybrid",
        query_kinds=("original", "rewrite", "hyde"),
        query_model="gpt-test-query",
        candidates=[],
        contexts=[(chunk, 0.91)],
        rerank_applied=True,
        rerank_provider="fake",
        rerank_model="gpt-test-reranker",
    )
    return Answer(
        answer=f"Answer to {question}",
        citations=[
            {
                "chunk_id": f"{citation_paper or paper}:0",
                "quote": "Evidence",
            }
        ],
        trace=trace,
    )


def _evaluation(
    context_relevance: float | None = 1.0,
    answer_relevance: float | None = 1.0,
    faithfulness: float | None = 1.0,
) -> Evaluation:
    return Evaluation(
        context_relevance=Metric(context_relevance, "context reason"),
        answer_relevance=Metric(answer_relevance, "answer reason"),
        faithfulness=Metric(faithfulness, "faithfulness reason"),
        contexts=[],
    )


def test_each_golden_case_runs_once_and_paper_ids_are_exact(monkeypatch, capsys):
    ask_calls = []
    judge_calls = []

    def fake_ask(question, *, query_mode, rerank_enabled):
        ask_calls.append(question)
        expected_paper = eval_cli.GOLDEN[question]
        # ``attention`` used to count the ``flash_attention`` citation because
        # the old CLI used substring matching. The exact flash-attention case
        # must still pass while the plain-attention case must not.
        actual_paper = (
            "flash_attention" if expected_paper == "attention" else expected_paper
        )
        return _answer(question, actual_paper)

    def fake_evaluate(question, answer, contexts, *, refused):
        assert refused is False
        judge_calls.append((question, answer, [chunk.id for chunk in contexts]))
        return _evaluation()

    monkeypatch.setattr(eval_cli, "ask", fake_ask)
    monkeypatch.setattr(eval_cli.evals, "evaluate", fake_evaluate)
    monkeypatch.setattr(sys, "argv", ["research-rag eval"])

    eval_cli.main()

    expected_questions = list(eval_cli.GOLDEN)
    assert ask_calls == expected_questions
    assert Counter(ask_calls) == Counter(
        {question: 1 for question in expected_questions}
    )
    assert [call[0] for call in judge_calls] == expected_questions

    output = capsys.readouterr().out
    assert "final-context paper hit rate: 9/10" in output
    assert "verified expected-paper citation rate: 9/10" in output


def test_metrics_are_aggregated_and_printed(monkeypatch, capsys):
    golden = {"First question?": "first", "Second question?": "second"}
    scores = {
        "First question?": (0.25, 0.5, None),
        "Second question?": (0.75, 1.0, 0.5),
    }
    judge_calls = []

    monkeypatch.setattr(eval_cli, "GOLDEN", golden)
    monkeypatch.setattr(
        eval_cli,
        "ask",
        lambda question, **kwargs: _answer(question, golden[question]),
    )

    def fake_evaluate(question, answer, contexts, *, refused):
        assert refused is False
        judge_calls.append((question, answer, [chunk.paper for chunk in contexts]))
        return _evaluation(*scores[question])

    monkeypatch.setattr(eval_cli.evals, "evaluate", fake_evaluate)
    monkeypatch.setattr(sys, "argv", ["research-rag eval"])

    eval_cli.main()

    assert judge_calls == [
        ("First question?", "Answer to First question?", ["first"]),
        ("Second question?", "Answer to Second question?", ["second"]),
    ]
    output = capsys.readouterr().out
    assert "mean context relevance: 0.50" in output
    assert "mean answer relevance: 0.75" in output
    assert "mean faithfulness: 0.50" in output


def test_skip_judge_avoids_judge_calls(monkeypatch, capsys):
    question = "Offline question?"
    monkeypatch.setattr(eval_cli, "GOLDEN", {question: "offline"})
    monkeypatch.setattr(
        eval_cli,
        "ask",
        lambda received, **kwargs: _answer(received, "offline"),
    )

    def unexpected_judge(*args, **kwargs):
        raise AssertionError("--skip-judge must not call the judge")

    monkeypatch.setattr(eval_cli.evals, "evaluate", unexpected_judge)
    monkeypatch.setattr(sys, "argv", ["research-rag eval", "--skip-judge"])

    eval_cli.main()

    output = capsys.readouterr().out
    assert "mean context relevance" not in output
    assert "mean answer relevance" not in output
    assert "mean faithfulness" not in output


def test_pipeline_errors_count_as_misses_in_deterministic_rates(monkeypatch, capsys):
    golden = {"Successful question?": "success", "Broken question?": "broken"}
    monkeypatch.setattr(eval_cli, "GOLDEN", golden)

    def fake_ask(question, **kwargs):
        if question == "Broken question?":
            raise RuntimeError("provider failed")
        return _answer(question, "success")

    monkeypatch.setattr(eval_cli, "ask", fake_ask)
    monkeypatch.setattr(sys, "argv", ["research-rag eval", "--skip-judge"])

    eval_cli.main()

    output = capsys.readouterr().out
    assert "final-context paper hit rate: 1/2" in output
    assert "verified expected-paper citation rate: 1/2" in output
    assert "pipeline errors (counted as misses): 1/2" in output


def test_json_writes_a_valid_report(monkeypatch, tmp_path, capsys):
    question = "Serializable question?"
    report_path = tmp_path / "nested" / "report.json"
    ask_calls = []

    monkeypatch.setattr(eval_cli, "GOLDEN", {question: "serial"})
    monkeypatch.setenv("RAG_VECTOR_BACKEND", "qdrant")
    monkeypatch.setattr(
        eval_cli.store,
        "readiness",
        lambda _settings: {
            "ready": True,
            "backend": "qdrant",
            "status": "green",
            "chunks": 12,
            "dimension": 3,
            "build_id": "a" * 32,
            "collection": "research-rag",
        },
    )

    def fake_ask(received, *, query_mode, rerank_enabled):
        ask_calls.append((received, query_mode, rerank_enabled))
        return _answer(received, "serial")

    monkeypatch.setattr(eval_cli, "ask", fake_ask)
    monkeypatch.setattr(
        eval_cli.evals,
        "evaluate",
        lambda *args, **kwargs: _evaluation(0.4, 0.6, 0.8),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "research-rag eval",
            "--query-mode",
            "rewrite",
            "--no-rerank",
            "--json",
            str(report_path),
        ],
    )

    eval_cli.main()

    report = json.loads(report_path.read_text())
    assert ask_calls == [(question, "rewrite", False)]
    assert report["judge_model"] == eval_cli.evals.MODEL
    assert report["configuration"]["vector_backend"] == "qdrant"
    assert report["index"] == {
        "ready": True,
        "backend": "qdrant",
        "status": "green",
        "chunks": 12,
        "dimension": 3,
        "build_id": "a" * 32,
        "collection": "research-rag",
    }
    assert len(report["cases"]) == 1
    case = report["cases"][0]
    assert case["question"] == question
    assert case["expected_paper"] == "serial"
    assert case["paper_hit"] is True
    assert case["cited_hit"] is True
    assert case["context_ids"] == ["serial:0"]
    assert case["pipeline"] == {
        "query_mode": "hybrid",
        "query_kinds": ["original", "rewrite", "hyde"],
        "query_model": "gpt-test-query",
        "rerank_applied": True,
        "rerank_provider": "fake",
        "rerank_model": "gpt-test-reranker",
        "rerank_error": None,
        "expansion_status": {},
        "expansion_errors": [],
        "retrieval_error": None,
        "generation_error": None,
    }
    assert case["evaluation"]["context_relevance"]["score"] == 0.4
    assert case["evaluation"]["answer_relevance"]["score"] == 0.6
    assert case["evaluation"]["faithfulness"]["score"] == 0.8
    assert f"report: {report_path}" in capsys.readouterr().out


def test_pii_question_is_redacted_in_judge_stdout_and_json(
    monkeypatch, tmp_path, capsys
):
    raw_question = "What does researcher@example.com say about retrieval?"
    safe_question = "What does [EMAIL_1] say about retrieval?"
    report_path = tmp_path / "report.json"
    judged_questions = []
    monkeypatch.setattr(eval_cli, "GOLDEN", {raw_question: "private"})

    def fake_ask(received, **kwargs):
        assert received == raw_question
        answer = _answer(safe_question, "private")
        answer.answer = "Retrieval uses indexed evidence."
        return answer

    def fake_evaluate(question, answer, contexts, *, refused):
        assert refused is False
        judged_questions.append(question)
        return _evaluation()

    monkeypatch.setattr(eval_cli, "ask", fake_ask)
    monkeypatch.setattr(eval_cli.evals, "evaluate", fake_evaluate)
    monkeypatch.setattr(sys, "argv", ["research-rag eval", "--json", str(report_path)])

    eval_cli.main()

    output = capsys.readouterr().out
    report_text = report_path.read_text()
    assert judged_questions == [safe_question]
    assert raw_question not in output
    assert raw_question not in report_text
    assert "researcher@example.com" not in output
    assert "researcher@example.com" not in report_text
    assert json.loads(report_text)["cases"][0]["question"] == safe_question
