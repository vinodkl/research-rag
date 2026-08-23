"""The online pipeline: every step in order, on one screen.

    input guard -> rewrite/HyDE -> retrieve/fuse -> rerank -> generate -> output guard

Query rewriting and HyDE are recall tools: the original query always remains in
the search, and a hypothetical document is never evidence. Reranking is the
precision stage. The final contexts are retained in an internal trace so evals
judge the exact run that produced the answer.
"""

import math
from dataclasses import dataclass, field

from research_rag import generation as generate
from research_rag import guardrails as guards
from research_rag.models import Chunk
from research_rag.retrieval import query_planning as query
from research_rag.retrieval import reranker as rerank
from research_rag.retrieval import search as retrieval
from research_rag.retrieval import store
from research_rag.settings import get_settings


@dataclass
class PipelineTrace:
    sanitized_question: str
    redactions: tuple[str, ...]
    query_mode: str
    query_kinds: tuple[str, ...]
    query_model: str | None
    candidates: list[retrieval.Candidate] = field(repr=False)
    contexts: list[tuple[Chunk, float]] = field(repr=False)
    rerank_applied: bool
    rerank_provider: str
    rerank_model: str | None = None
    rerank_error: str | None = None
    expansion_status: dict[str, str] = field(default_factory=dict)
    expansion_errors: tuple[str, ...] = ()
    retrieval_error: str | None = None
    generation_error: str | None = None


@dataclass
class Answer:
    answer: str
    citations: list[dict] = field(default_factory=list)
    refused: bool = False
    unavailable: bool = False
    trace: PipelineTrace | None = field(default=None, repr=False)


def ask(
    question: str,
    *,
    query_mode: str | None = None,
    rerank_enabled: bool | None = None,
) -> Answer:
    settings = get_settings()
    # Guard 1: is the question safe and sane? This runs before every model call.
    if refusal := guards.check_question(question):
        return Answer(answer=refusal, refused=True)

    sanitized = guards.sanitize_question(question)
    mode = (query_mode or settings.query_mode).lower()
    if mode not in {"original", "rewrite", "hyde", "hybrid", "auto"}:
        mode = "auto"

    # Recall: search the original plus any safe, useful alternate query views.
    try:
        expansion = query.expand(sanitized.text, mode=mode)
    except Exception as exc:  # noqa: BLE001 - original-query fallback is mandatory
        selected = (
            "hybrid" if mode == "auto" and query.should_expand(sanitized.text) else mode
        )
        statuses = {
            strategy: "unavailable"
            if selected in {strategy, "hybrid"}
            else "not-requested"
            for strategy in ("rewrite", "hyde")
        }
        expansion = query.ExpansionResult(
            variants=[query.QueryVariant(sanitized.text, "original")],
            strategy_status=statuses,
            errors=(f"Query expansion failed ({type(exc).__name__})",),
            model=settings.query_model,
        )
    variants = expansion.variants

    retrieval_diagnostics: dict[str, str] = {}
    try:
        vector_store = store.load(settings)
        candidates = retrieval.retrieve(
            vector_store,
            variants,
            per_query_k=settings.per_query_k,
            candidate_k=settings.candidate_k,
            diagnostics=retrieval_diagnostics,
        )
    except Exception as exc:  # noqa: BLE001 - do not expose provider/index failures
        trace = _trace(
            sanitized,
            mode,
            expansion,
            [],
            [],
            None,
            retrieval_error=f"Retrieval failed ({type(exc).__name__})",
        )
        return Answer(
            answer="I could not retrieve evidence from the indexed papers.",
            refused=True,
            unavailable=True,
            trace=trace,
        )
    retrieval_error = retrieval_diagnostics.get("batch_error")

    # Guard 2: only real-query cosine scores can establish topic/evidence fit.
    guard_results = [(item.chunk, item.guard_score) for item in candidates]
    if refusal := guards.check_retrieval(guard_results):
        trace = _trace(
            sanitized,
            mode,
            expansion,
            candidates,
            [],
            None,
            retrieval_error=retrieval_error,
        )
        return Answer(answer=refusal, refused=True, trace=trace)

    # Precision: jointly rank and grade every real candidate against the question.
    context_k = settings.context_k
    if rerank_enabled is None:
        rerank_enabled = settings.rerank_enabled
    if rerank_enabled:
        try:
            provider_result = rerank.rerank(
                sanitized.text, retrieval.as_scored_chunks(candidates), top_k=context_k
            )
        except Exception as exc:  # noqa: BLE001 - retain guarded vector fallback
            provider_result = rerank.RerankResult(
                results=retrieval.as_scored_chunks(candidates)[:context_k],
                applied=False,
                provider="openai",
                error=f"OpenAI reranking failed ({type(exc).__name__})",
                model=settings.rerank_model,
            )
        if provider_result.applied:
            # The listwise judge read the original question and every real chunk
            # together. It may therefore rescue a chunk found only through HyDE.
            candidate_ids = {item.chunk.id for item in candidates}
            results = [
                (chunk, score)
                for chunk, score in provider_result.results
                if chunk.id in candidate_ids
                and math.isfinite(score)
                and score >= rerank.GRADE_SCORE["supporting"]
            ][:context_k]
            reranked = provider_result
        else:
            # Without a joint reranker, HyDE-only candidates cannot establish
            # relevance by themselves. Keep fused order, but require each final
            # context to have passed a real-query cosine floor.
            results = [
                (item.chunk, item.vector_score)
                for item in candidates
                if math.isfinite(item.guard_score)
                and item.guard_score >= guards.MIN_SCORE
            ][:context_k]
            reranked = rerank.RerankResult(
                results=results,
                applied=False,
                provider=provider_result.provider,
                error=provider_result.error,
                model=provider_result.model,
            )
    else:
        results = [
            (item.chunk, item.vector_score)
            for item in candidates
            if math.isfinite(item.guard_score) and item.guard_score >= guards.MIN_SCORE
        ][:context_k]
        reranked = rerank.RerankResult(
            results=results,
            applied=False,
            provider="disabled",
            error=None,
            model=None,
        )

    trace = _trace(
        sanitized,
        mode,
        expansion,
        candidates,
        results,
        reranked,
        retrieval_error=retrieval_error,
    )
    if not results:
        return Answer(
            answer="I could not find enough evidence in the indexed papers to answer that.",
            refused=True,
            trace=trace,
        )

    # Generate: answer + citations, from the passages only.
    try:
        raw = generate.generate(sanitized.text, results)
    except Exception as exc:  # noqa: BLE001 - model/JSON failures become safe refusals
        trace.generation_error = f"Generation failed ({type(exc).__name__})"
        return Answer(
            answer="I could not produce a usable answer from the indexed papers.",
            refused=True,
            unavailable=True,
            trace=trace,
        )

    # Guard 3: validate output and citations against only the final contexts.
    if not isinstance(raw, dict):
        trace.generation_error = "Generation returned an invalid payload"
        return Answer(
            answer="I could not produce a usable answer from the indexed papers.",
            refused=True,
            unavailable=True,
            trace=trace,
        )
    if refusal := guards.check_output(raw.get("answer", "")):
        return Answer(answer=refusal, refused=True, trace=trace)
    citations = guards.check_citations(raw.get("citations", []), results)
    if not citations:
        return Answer(
            answer="I could not produce an answer with verifiable citations, so I would rather refuse than guess.",
            refused=True,
            trace=trace,
        )
    return Answer(answer=raw["answer"], citations=citations, trace=trace)


def _trace(
    sanitized: guards.SanitizedQuestion,
    mode: str,
    expansion: query.ExpansionResult,
    candidates: list[retrieval.Candidate],
    contexts: list[tuple[Chunk, float]],
    reranked: rerank.RerankResult | None,
    *,
    retrieval_error: str | None = None,
) -> PipelineTrace:
    return PipelineTrace(
        sanitized_question=sanitized.text,
        redactions=sanitized.redactions,
        query_mode=mode,
        # Keep provenance, but not hypothetical text, out of traces/logs.
        query_kinds=tuple(variant.kind for variant in expansion.variants),
        query_model=expansion.model,
        candidates=candidates,
        contexts=contexts,
        rerank_applied=bool(reranked and reranked.applied),
        rerank_provider=reranked.provider if reranked else "not-run",
        rerank_model=reranked.model if reranked else None,
        rerank_error=reranked.error if reranked else None,
        expansion_status=dict(expansion.strategy_status),
        expansion_errors=tuple(expansion.errors),
        retrieval_error=retrieval_error,
    )
