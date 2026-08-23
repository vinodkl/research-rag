"""Compose the online RAG stages in one visible sequence.

    input guard -> rewrite/HyDE -> retrieve/fuse -> rerank -> generate -> output guard

The original query always remains in search, HyDE is never evidence, and the
final contexts are retained so evals judge the exact run that made the answer.
"""

import math
from dataclasses import dataclass, field

from research_rag import generation, guardrails
from research_rag.models import Chunk
from research_rag.retrieval import query_planning, reranker, search
from research_rag.retrieval import store as vector_store
from research_rag.settings import Settings, get_settings


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
    mode = (query_mode or settings.query_mode).lower()
    if mode not in query_planning.MODES:
        mode = "auto"

    # 2. RECALL: search the original question plus optional rewrites and HyDE.
    expansion = query_planning.expand(sanitized.text, mode=mode, settings=settings)

    diagnostics: dict[str, str] = {}
    try:
        backend = vector_store.load(settings)
        candidates = search.retrieve(
            backend,
            expansion.variants,
            per_query_k=settings.per_query_k,
            candidate_k=settings.candidate_k,
            diagnostics=diagnostics,
        )
    except Exception as exc:  # noqa: BLE001 - do not expose provider/index failures
        error = f"Retrieval failed ({type(exc).__name__})"
        trace = _make_trace(sanitized, mode, expansion, [], [], None, error)
        return _unavailable(
            "I could not retrieve evidence from the indexed papers.", trace
        )
    retrieval_error = diagnostics.get("batch_error")

    # 3. EVIDENCE GUARD: HyDE alone cannot prove that evidence is relevant.
    if refusal := guardrails.check_retrieval(
        [(candidate.chunk, candidate.real_query_score) for candidate in candidates]
    ):
        trace = _make_trace(
            sanitized, mode, expansion, candidates, [], None, retrieval_error
        )
        return Answer(answer=refusal, refused=True, trace=trace)

    # 4. PRECISION: rerank broad candidates into the final answer contexts.
    use_reranker = settings.rerank_enabled if rerank_enabled is None else rerank_enabled
    final_contexts, rerank_result = _select_contexts(
        sanitized.text, candidates, use_reranker, settings
    )
    trace = _make_trace(
        sanitized,
        mode,
        expansion,
        candidates,
        final_contexts,
        rerank_result,
        retrieval_error,
    )
    if not final_contexts:
        return Answer(
            answer="I could not find enough evidence in the indexed papers to answer that.",
            refused=True,
            trace=trace,
        )

    # 5. GENERATE + OUTPUT GUARD: answer only from the selected real contexts.
    try:
        generated = generation.generate(
            sanitized.text, final_contexts, settings=settings
        )
    except Exception as exc:  # noqa: BLE001 - return a safe operational refusal
        trace.generation_error = f"Generation failed ({type(exc).__name__})"
        return _unavailable(
            "I could not produce a usable answer from the indexed papers.", trace
        )

    if not isinstance(generated, dict):
        trace.generation_error = "Generation returned an invalid payload"
        return _unavailable(
            "I could not produce a usable answer from the indexed papers.", trace
        )
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


def _select_contexts(
    question: str,
    candidates: list[search.Candidate],
    enabled: bool,
    settings: Settings,
) -> tuple[list[tuple[Chunk, float]], reranker.RerankResult]:
    """Choose answer contexts with reranking or a guarded vector fallback."""

    fallback = [
        (candidate.chunk, candidate.best_similarity)
        for candidate in candidates
        if math.isfinite(candidate.real_query_score)
        and candidate.real_query_score >= guardrails.MIN_SCORE
    ][: settings.context_k]
    if not enabled:
        return fallback, reranker.RerankResult(
            results=fallback,
            applied=False,
            provider="disabled",
        )

    model_result = reranker.rerank(
        question,
        search.as_scored_chunks(candidates),
        top_k=settings.context_k,
        settings=settings,
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

    return fallback, reranker.RerankResult(
        results=fallback,
        applied=False,
        provider=model_result.provider,
        error=model_result.error,
        model=model_result.model,
    )


def _unavailable(message: str, trace: PipelineTrace) -> Answer:
    return Answer(
        answer=message,
        refused=True,
        unavailable=True,
        trace=trace,
    )


def _make_trace(
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
