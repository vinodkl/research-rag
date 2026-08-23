"""End-to-end RAG evaluation over questions with known source papers.

Each question runs through ``ask`` exactly once. Deterministic paper-hit and
verified-citation checks use that run's final contexts, then up to one structured
judge call scores final contexts and, for non-refused answers, answer relevance
and faithfulness.

Run:
    research-rag eval
    research-rag eval --skip-judge
    research-rag eval --json data/eval/latest.json
"""

import argparse
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from research_rag import guardrails as guards
from research_rag.evaluation import judge as evals
from research_rag.models import Chunk
from research_rag.pipeline import Answer, PipelineTrace, ask
from research_rag.retrieval import store
from research_rag.settings import Settings, get_settings

REPORT_SCHEMA_VERSION = 1
_METRIC_NAMES = ("context_relevance", "answer_relevance", "faithfulness")
_Record = dict[str, Any]


@dataclass(frozen=True)
class _RunSummary:
    """Deterministic counts shared by terminal output and JSON reports."""

    case_count: int
    completed_count: int
    paper_hits: int
    cited_hits: int

    @property
    def pipeline_error_count(self) -> int:
        return self.case_count - self.completed_count


def load_golden(path: Path) -> dict[str, str]:
    """Load visible YAML cases instead of hiding the regression set in code."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("golden questions must be a non-empty YAML list")
    cases: dict[str, str] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each golden case must be an object")
        question = item.get("question")
        paper = item.get("expected_paper")
        if not isinstance(question, str) or not isinstance(paper, str):
            raise ValueError("golden cases need question and expected_paper strings")
        if question in cases:
            raise ValueError("golden questions must be unique")
        cases[question] = paper
    return cases


# Kept as a module value so a lesson or unit test can replace the tiny dataset.
# A wheel installed outside a checkout has no implicit corpus configuration;
# deferring the missing-file error keeps `research-rag eval --help` usable.
_DEFAULT_GOLDEN_PATH = get_settings().golden_questions_path
GOLDEN = load_golden(_DEFAULT_GOLDEN_PATH) if _DEFAULT_GOLDEN_PATH.is_file() else {}


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    args = _parse_args(argv)
    cases = _load_cases(args, settings)
    records = _evaluate_cases(cases, args)
    summary = _summarize(records)
    _print_summary(records, summary, judge_enabled=not args.skip_judge)
    if args.json:
        _write_report(args.json, records, summary, args, settings)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="research-rag eval", description=__doc__)
    parser.add_argument(
        "--limit", type=int, help="evaluate only the first N golden questions"
    )
    parser.add_argument(
        "--skip-judge", action="store_true", help="run only deterministic checks"
    )
    parser.add_argument(
        "--json", type=Path, help="also write a machine-readable report"
    )
    parser.add_argument(
        "--golden",
        type=Path,
        help="override RAG_GOLDEN_PATH for this run",
    )
    parser.add_argument(
        "--query-mode",
        choices=("original", "rewrite", "hyde", "hybrid", "auto"),
        default=None,
        help="override RAG_QUERY_MODE for this run",
    )
    parser.add_argument(
        "--no-rerank", action="store_true", help="disable reranking for this run"
    )
    return parser.parse_args(argv)


def _load_cases(
    args: argparse.Namespace,
    settings: Settings,
) -> list[tuple[str, str]]:
    if args.golden:
        golden = load_golden(args.golden.expanduser().resolve())
    else:
        golden = GOLDEN or load_golden(settings.golden_questions_path)
    cases = list(golden.items())
    if args.limit is not None:
        cases = cases[: max(args.limit, 0)]
    return cases


def _evaluate_cases(
    cases: list[tuple[str, str]],
    args: argparse.Namespace,
) -> list[_Record]:
    records: list[_Record] = []
    print(f"{'question':50s} paper  cited  context  answer  faithful")
    for question, expected_paper in cases:
        record, judged = _evaluate_case(
            question,
            expected_paper,
            query_mode=args.query_mode,
            rerank_enabled=False if args.no_rerank else None,
            judge_enabled=not args.skip_judge,
        )
        records.append(record)
        _print_case(record, judged)
    return records


def _evaluate_case(
    question: str,
    expected_paper: str,
    *,
    query_mode: str | None,
    rerank_enabled: bool | None,
    judge_enabled: bool,
) -> tuple[_Record, evals.Evaluation | None]:
    safe_question = guards.sanitize_question(question).text
    try:
        answer = ask(
            question,
            query_mode=query_mode,
            rerank_enabled=rerank_enabled,
        )
    except Exception as exc:  # noqa: BLE001 - one failed case must not end the run
        return (
            {
                "question": safe_question,
                "expected_paper": expected_paper,
                "error": type(exc).__name__,
            },
            None,
        )

    evaluated_question = (
        answer.trace.sanitized_question if answer.trace else safe_question
    )
    contexts = answer.trace.contexts if answer.trace else []
    context_chunks = [chunk for chunk, _ in contexts]
    paper_hit = any(chunk.paper == expected_paper for chunk in context_chunks)
    cited_hit = any(
        citation.get("chunk_id", "").split(":", 1)[0] == expected_paper
        for citation in answer.citations
    )
    judged = (
        evals.evaluate(
            evaluated_question,
            answer.answer,
            context_chunks,
            refused=answer.refused,
        )
        if judge_enabled
        else None
    )
    return (
        _case_record(
            answer,
            evaluated_question,
            expected_paper,
            context_chunks,
            paper_hit=paper_hit,
            cited_hit=cited_hit,
            judged=judged,
        ),
        judged,
    )


def _case_record(
    answer: Answer,
    evaluated_question: str,
    expected_paper: str,
    context_chunks: list[Chunk],
    *,
    paper_hit: bool,
    cited_hit: bool,
    judged: evals.Evaluation | None,
) -> _Record:
    return {
        "question": evaluated_question,
        "expected_paper": expected_paper,
        "paper_hit": paper_hit,
        "cited_hit": cited_hit,
        "refused": answer.refused,
        "unavailable": answer.unavailable,
        "answer": answer.answer,
        "citations": answer.citations,
        "context_ids": [chunk.id for chunk in context_chunks],
        "pipeline": _pipeline_record(answer.trace),
        "evaluation": asdict(judged) if judged else None,
    }


def _pipeline_record(trace: PipelineTrace | None) -> _Record:
    if trace is None:
        return {
            "query_mode": None,
            "query_kinds": [],
            "query_model": None,
            "rerank_applied": False,
            "rerank_provider": None,
            "rerank_model": None,
            "rerank_error": None,
            "expansion_status": {},
            "expansion_errors": [],
            "retrieval_error": None,
            "generation_error": None,
        }
    return {
        "query_mode": trace.query_mode,
        "query_kinds": list(trace.query_kinds),
        "query_model": trace.query_model,
        "rerank_applied": trace.rerank_applied,
        "rerank_provider": trace.rerank_provider,
        "rerank_model": trace.rerank_model,
        "rerank_error": trace.rerank_error,
        "expansion_status": trace.expansion_status,
        "expansion_errors": list(trace.expansion_errors),
        "retrieval_error": trace.retrieval_error,
        "generation_error": trace.generation_error,
    }


def _print_case(record: _Record, judged: evals.Evaluation | None) -> None:
    question = record["question"]
    if "error" in record:
        print(f"{question[:49]:50s} ERROR ({record['error']})")
        return
    print(
        f"{question[:49]:50s} "
        f"{_yes(record['paper_hit']):6s} {_yes(record['cited_hit']):6s} "
        f"{_metric(judged, 'context_relevance'):7s} "
        f"{_metric(judged, 'answer_relevance'):7s} "
        f"{_metric(judged, 'faithfulness'):8s}"
    )


def _summarize(records: list[_Record]) -> _RunSummary:
    completed_count = sum("error" not in record for record in records)
    return _RunSummary(
        case_count=len(records),
        completed_count=completed_count,
        paper_hits=sum(bool(record.get("paper_hit", False)) for record in records),
        cited_hits=sum(bool(record.get("cited_hit", False)) for record in records),
    )


def _print_summary(
    records: list[_Record],
    summary: _RunSummary,
    *,
    judge_enabled: bool,
) -> None:
    print(f"\nfinal-context paper hit rate: {summary.paper_hits}/{summary.case_count}")
    print(
        f"verified expected-paper citation rate: "
        f"{summary.cited_hits}/{summary.case_count}"
    )
    if summary.pipeline_error_count:
        print(
            "pipeline errors (counted as misses): "
            f"{summary.pipeline_error_count}/{summary.case_count}"
        )
    if judge_enabled:
        for name in _METRIC_NAMES:
            print(
                f"mean {name.replace('_', ' ')}: {_number(_mean_metric(records, name))}"
            )


def _mean_metric(records: list[_Record], name: str) -> float | None:
    scores: list[float] = [
        record["evaluation"][name]["score"]
        for record in records
        if "error" not in record
        and record["evaluation"]
        and record["evaluation"][name]["score"] is not None
    ]
    return sum(scores) / len(scores) if scores else None


def _write_report(
    path: Path,
    records: list[_Record],
    summary: _RunSummary,
    args: argparse.Namespace,
    settings: Settings,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = _build_report(records, summary, args, settings)
    _atomic_json(path, report)
    print(f"report: {path}")


def _build_report(
    records: list[_Record],
    summary: _RunSummary,
    args: argparse.Namespace,
    settings: Settings,
) -> _Record:
    total = summary.case_count
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "judge_model": None if args.skip_judge else settings.evaluation_model,
        "configuration": {
            "query_mode": args.query_mode or settings.query_mode,
            "rerank_enabled": not args.no_rerank and settings.rerank_enabled,
            "vector_backend": settings.vector_backend,
            "embedding_model": settings.embedding_model,
            "generation_model": settings.generation_model,
        },
        "index": store.readiness(settings),
        "summary": {
            "case_count": total,
            "completed_count": summary.completed_count,
            "pipeline_error_count": summary.pipeline_error_count,
            "paper_hit_rate": summary.paper_hits / total if total else None,
            "verified_citation_rate": summary.cited_hits / total if total else None,
        },
        "cases": records,
    }


def _atomic_json(path: Path, value: dict) -> None:
    """Never leave a half-written report for CI or a later lesson."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _yes(value: bool) -> str:
    return "yes" if value else "NO"


def _metric(evaluation: evals.Evaluation | None, name: str) -> str:
    if evaluation is None:
        return "-"
    return _number(getattr(evaluation, name).score)


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


if __name__ == "__main__":
    main()
