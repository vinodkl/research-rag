"""The online pipeline: every stage in order, on one screen.

    input guard -> rewrite/HyDE -> retrieve/fuse -> rerank -> generate -> output guard

Query rewriting and HyDE are recall tools: the original query always remains in
the search, and a hypothetical document is never evidence. Reranking is the
precision stage. The final contexts are retained in an internal trace so evals
judge the exact run that produced the answer.
"""

import math
from dataclasses import dataclass, field

from research_rag import generation, guardrails
from research_rag.models import Chunk
from research_rag.retrieval import query_planning, reranker, search
from research_rag.retrieval import store as vector_store
from research_rag.settings import Settings, get_settings

QUERY_MODES = {"original", "rewrite", "hyde", "hybrid", "auto"}


@dataclass
class PipelineTrace:
    sanitized_question: str
    redactions: tuple[str, ...]
    query_mode: str
    query_kinds: tuple[str, ...]
    query_model: str | None
    candidates: list[search.Candidate] = field(repr=False)
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
    """Answer one question by composing the visible RAG stages below."""

    settings = get_settings()

    # 1. INPUT GUARD: reject unsafe input before any model or database call.
    if refusal := guardrails.check_question(question):
        return Answer(answer=refusal, refused=True)

    sanitized = guardrails.sanitize_question(question)
    mode = _resolve_query_mode(query_mode or settings.query_mode)

    # 2. RECALL: search the original question plus optional rewrites and HyDE.
    expansion = _expand_queries(sanitized.text, mode, settings)
    try:
        candidates, retrieval_error = _retrieve_candidates(expansion.variants, settings)
    except Exception as exc:  # noqa: BLE001 - do not expose provider/index failures
        trace = _make_trace(
            sanitized=sanitized,
            mode=mode,
            expansion=expansion,
            candidates=[],
            contexts=[],
            rerank_result=None,
            retrieval_error=f"Retrieval failed ({type(exc).__name__})",
        )
        return Answer(
            answer="I could not retrieve evidence from the indexed papers.",
            refused=True,
            unavailable=True,
            trace=trace,
        )

    # 3. EVIDENCE GUARD: HyDE alone cannot prove that evidence is relevant.
    evidence_scores = [
        (candidate.chunk, candidate.real_query_score) for candidate in candidates
    ]
    if refusal := guardrails.check_retrieval(evidence_scores):
        trace = _make_trace(
            sanitized=sanitized,
            mode=mode,
            expansion=expansion,
            candidates=candidates,
            contexts=[],
            rerank_result=None,
            retrieval_error=retrieval_error,
        )
        return Answer(answer=refusal, refused=True, trace=trace)

    # 4. PRECISION: rerank broad candidates into the final answer contexts.
    use_reranker = settings.rerank_enabled if rerank_enabled is None else rerank_enabled
    final_contexts, rerank_result = _select_contexts(
        sanitized.text, candidates, use_reranker, settings
    )
    trace = _make_trace(
        sanitized=sanitized,
        mode=mode,
        expansion=expansion,
        candidates=candidates,
        contexts=final_contexts,
        rerank_result=rerank_result,
        retrieval_error=retrieval_error,
    )
    if not final_contexts:
        return Answer(
            answer="I could not find enough evidence in the indexed papers to answer that.",
            refused=True,
            trace=trace,
        )

    # 5. GENERATE + OUTPUT GUARD: answer only from the selected real contexts.
    return _generate_answer(sanitized.text, final_contexts, trace, settings)


def _resolve_query_mode(mode: str) -> str:
    normalized = mode.lower()
    return normalized if normalized in QUERY_MODES else "auto"


def _expand_queries(
    question: str, mode: str, settings: Settings
) -> query_planning.ExpansionResult:
    """Plan extra search views; fall back to the original question on failure."""

    try:
        return query_planning.expand(question, mode=mode, settings=settings)
    except Exception as exc:  # noqa: BLE001 - original-query fallback is mandatory
        selected = (
            "hybrid"
            if mode == "auto" and query_planning.should_expand(question)
            else mode
        )
        statuses = {
            strategy: "unavailable"
            if selected in {strategy, "hybrid"}
            else "not-requested"
            for strategy in ("rewrite", "hyde")
        }
        return query_planning.ExpansionResult(
            variants=[query_planning.QueryVariant(question, "original")],
            strategy_status=statuses,
            errors=(f"Query expansion failed ({type(exc).__name__})",),
            model=settings.query_model,
        )


def _retrieve_candidates(
    variants: list[query_planning.QueryVariant], settings: Settings
) -> tuple[list[search.Candidate], str | None]:
    diagnostics: dict[str, str] = {}
    backend = vector_store.load(settings)
    candidates = search.retrieve(
        backend,
        variants,
        per_query_k=settings.per_query_k,
        candidate_k=settings.candidate_k,
        diagnostics=diagnostics,
    )
    return candidates, diagnostics.get("batch_error")


def _select_contexts(
    question: str,
    candidates: list[search.Candidate],
    enabled: bool,
    settings: Settings,
) -> tuple[list[tuple[Chunk, float]], reranker.RerankResult]:
    """Choose answer contexts with reranking or a guarded vector fallback."""

    if not enabled:
        contexts = _real_query_contexts(candidates, settings.context_k)
        return contexts, reranker.RerankResult(
            results=contexts,
            applied=False,
            provider="disabled",
        )

    scored_candidates = search.as_scored_chunks(candidates)
    try:
        model_result = reranker.rerank(
            question,
            scored_candidates,
            top_k=settings.context_k,
            settings=settings,
        )
    except Exception as exc:  # noqa: BLE001 - retain guarded vector fallback
        model_result = reranker.RerankResult(
            results=scored_candidates[: settings.context_k],
            applied=False,
            provider="openai",
            error=f"OpenAI reranking failed ({type(exc).__name__})",
            model=settings.rerank_model,
        )

    if model_result.applied:
        candidate_ids = {candidate.chunk.id for candidate in candidates}
        contexts = [
            (chunk, score)
            for chunk, score in model_result.results
            if chunk.id in candidate_ids
            and math.isfinite(score)
            and score >= reranker.GRADE_SCORE["supporting"]
        ][: settings.context_k]
        return contexts, model_result

    contexts = _real_query_contexts(candidates, settings.context_k)
    fallback_result = reranker.RerankResult(
        results=contexts,
        applied=False,
        provider=model_result.provider,
        error=model_result.error,
        model=model_result.model,
    )
    return contexts, fallback_result


def _real_query_contexts(
    candidates: list[search.Candidate], limit: int
) -> list[tuple[Chunk, float]]:
    """Keep fused order, but require evidence found without synthetic HyDE."""

    return [
        (candidate.chunk, candidate.best_similarity)
        for candidate in candidates
        if math.isfinite(candidate.real_query_score)
        and candidate.real_query_score >= guardrails.MIN_SCORE
    ][:limit]


def _generate_answer(
    question: str,
    final_contexts: list[tuple[Chunk, float]],
    trace: PipelineTrace,
    settings: Settings,
) -> Answer:
    try:
        generated = generation.generate(
            question,
            final_contexts,
            settings=settings,
        )
    except Exception as exc:  # noqa: BLE001 - return a safe operational refusal
        trace.generation_error = f"Generation failed ({type(exc).__name__})"
        return _generation_failure(trace)

    if not isinstance(generated, dict):
        trace.generation_error = "Generation returned an invalid payload"
        return _generation_failure(trace)
    if refusal := guardrails.check_output(generated.get("answer", "")):
        return Answer(answer=refusal, refused=True, trace=trace)
    citations = guardrails.check_citations(
        generated.get("citations", []), final_contexts
    )
    if not citations:
        return Answer(
            answer="I could not produce an answer with verifiable citations, so I would rather refuse than guess.",
            refused=True,
            trace=trace,
        )
    return Answer(answer=generated["answer"], citations=citations, trace=trace)


def _generation_failure(trace: PipelineTrace) -> Answer:
    return Answer(
        answer="I could not produce a usable answer from the indexed papers.",
        refused=True,
        unavailable=True,
        trace=trace,
    )


def _make_trace(
    *,
    sanitized: guardrails.SanitizedQuestion,
    mode: str,
    expansion: query_planning.ExpansionResult,
    candidates: list[search.Candidate],
    contexts: list[tuple[Chunk, float]],
    rerank_result: reranker.RerankResult | None,
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
        rerank_applied=bool(rerank_result and rerank_result.applied),
        rerank_provider=rerank_result.provider if rerank_result else "not-run",
        rerank_model=rerank_result.model if rerank_result else None,
        rerank_error=rerank_result.error if rerank_result else None,
        expansion_status=dict(expansion.strategy_status),
        expansion_errors=tuple(expansion.errors),
        retrieval_error=retrieval_error,
    )
