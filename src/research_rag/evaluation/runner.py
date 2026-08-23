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
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from research_rag import guardrails as guards
from research_rag.evaluation import judge as evals
from research_rag.pipeline import ask
from research_rag.retrieval import store
from research_rag.settings import get_settings

REPORT_SCHEMA_VERSION = 1


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
    args = parser.parse_args(argv)

    if args.golden:
        golden = load_golden(args.golden.expanduser().resolve())
    else:
        golden = GOLDEN or load_golden(settings.golden_questions_path)
    cases = list(golden.items())
    if args.limit is not None:
        cases = cases[: max(args.limit, 0)]

    records: list[dict[str, Any]] = []
    print(f"{'question':50s} paper  cited  context  answer  faithful")
    for question, expected_paper in cases:
        safe_question = guards.sanitize_question(question).text
        try:
            answer = ask(
                question,
                query_mode=args.query_mode,
                rerank_enabled=False if args.no_rerank else None,
            )
        except Exception as exc:  # noqa: BLE001 - one failed case must not end the run
            records.append(
                {
                    "question": safe_question,
                    "expected_paper": expected_paper,
                    "error": type(exc).__name__,
                }
            )
            print(f"{safe_question[:49]:50s} ERROR ({type(exc).__name__})")
            continue

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

        judged = None
        if not args.skip_judge:
            judged = evals.evaluate(
                evaluated_question,
                answer.answer,
                context_chunks,
                refused=answer.refused,
            )

        records.append(
            {
                "question": evaluated_question,
                "expected_paper": expected_paper,
                "paper_hit": paper_hit,
                "cited_hit": cited_hit,
                "refused": answer.refused,
                "unavailable": answer.unavailable,
                "answer": answer.answer,
                "citations": answer.citations,
                "context_ids": [chunk.id for chunk in context_chunks],
                "pipeline": {
                    "query_mode": answer.trace.query_mode if answer.trace else None,
                    "query_kinds": list(answer.trace.query_kinds)
                    if answer.trace
                    else [],
                    "query_model": answer.trace.query_model if answer.trace else None,
                    "rerank_applied": answer.trace.rerank_applied
                    if answer.trace
                    else False,
                    "rerank_provider": answer.trace.rerank_provider
                    if answer.trace
                    else None,
                    "rerank_model": answer.trace.rerank_model if answer.trace else None,
                    "rerank_error": answer.trace.rerank_error if answer.trace else None,
                    "expansion_status": answer.trace.expansion_status
                    if answer.trace
                    else {},
                    "expansion_errors": list(answer.trace.expansion_errors)
                    if answer.trace
                    else [],
                    "retrieval_error": answer.trace.retrieval_error
                    if answer.trace
                    else None,
                    "generation_error": answer.trace.generation_error
                    if answer.trace
                    else None,
                },
                "evaluation": asdict(judged) if judged else None,
            }
        )

        print(
            f"{evaluated_question[:49]:50s} "
            f"{_yes(paper_hit):6s} {_yes(cited_hit):6s} "
            f"{_metric(judged, 'context_relevance'):7s} "
            f"{_metric(judged, 'answer_relevance'):7s} "
            f"{_metric(judged, 'faithfulness'):8s}"
        )

    completed = [record for record in records if "error" not in record]
    total = len(records)
    failures = total - len(completed)
    paper_hits = sum(bool(record.get("paper_hit", False)) for record in records)
    cited_hits = sum(bool(record.get("cited_hit", False)) for record in records)
    print(f"\nfinal-context paper hit rate: {paper_hits}/{total}")
    print(f"verified expected-paper citation rate: {cited_hits}/{total}")
    if failures:
        print(f"pipeline errors (counted as misses): {failures}/{total}")
    if not args.skip_judge:
        for name in ("context_relevance", "answer_relevance", "faithfulness"):
            scores: list[float] = [
                record["evaluation"][name]["score"]
                for record in completed
                if record["evaluation"]
                and record["evaluation"][name]["score"] is not None
            ]
            value = sum(scores) / len(scores) if scores else None
            print(f"mean {name.replace('_', ' ')}: {_number(value)}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        successful = [record for record in records if "error" not in record]
        report = {
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
                "case_count": len(records),
                "completed_count": len(successful),
                "pipeline_error_count": len(records) - len(successful),
                "paper_hit_rate": paper_hits / total if total else None,
                "verified_citation_rate": cited_hits / total if total else None,
            },
            "cases": records,
        }
        _atomic_json(args.json, report)
        print(f"report: {args.json}")


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
