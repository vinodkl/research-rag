# Evaluation

`research-rag eval` runs every YAML case through the real pipeline once. All
metrics for that case use the answer and final contexts from that exact run.
It also uses the configured vector backend; evaluation never substitutes FAISS
when a selected Qdrant service is unavailable.

The regression set is deliberately visible in `config/golden_questions.yaml`:

```yaml
- question: How does LoRA reduce the number of trainable parameters?
  expected_paper: lora
```

The paper ID is a retrieval expectation, not an instruction to use LoRA in the
application.

## Deterministic checks

- **Final-context paper hit:** at least one final chunk came from the expected
  paper.
- **Verified expected-paper citation:** at least one mechanically verified
  citation came from that paper.

Run these without an evaluator model:

```bash
research-rag eval --skip-judge
```

## Judge metrics

The OpenAI judge makes at most one structured call per case with final contexts.
Python computes the aggregates so the denominators remain explicit.

- **Context relevance:** every final context receives a 0–1 usefulness score;
  the metric is their mean. Contexts are scored even if generation refused.
- **Answer relevance:** 0–1 directness and completeness with respect to the
  question, considered separately from grounding.
- **Faithfulness:** supported atomic claims divided by valid atomic claims using
  only final contexts. A refusal has no faithfulness score.

## Reproducible reports

```bash
research-rag eval --json data/eval/latest.json
research-rag eval --limit 3 --query-mode original --no-rerank
research-rag eval --golden path/to/cases.yaml
```

Reports are atomically written and include:

- report schema version and UTC creation time;
- safe model and retrieval configuration;
- selected vector backend and non-sensitive index readiness metadata;
- aggregate hit/error rates;
- per-case answer, verified citations, context IDs, scores, reasons, and safe
  fallback diagnostics.

Judge scores vary across runs and models. Compare distributions and regressions,
retain representative human review, and never use one score as proof of quality.

When comparing FAISS with Qdrant, hold the corpus, chunker, embedding model,
query mode, reranker, and golden cases constant. Ingest both backends from the
same corpus revision, then label and retain each JSON report. Similarity should
be comparable because both adapters use cosine distance, but exact ties and
floating-point order may differ. Treat a backend change like any other retrieval
release: inspect per-case regressions as well as averages.
