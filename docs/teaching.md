# Teaching guide

Teach this project as one visible question pipeline, not as a tour of every
file. Start with the local FAISS backend so the class can focus on retrieval
quality. Introduce Qdrant only after students understand the algorithm; it
changes storage and operations, not the RAG logic.

## The lesson in one sentence

RAG first finds a broad set of real passages, then selects the most useful
ones, then writes an answer that can cite only those passages.

## Recommended reading order

1. `models.py` — the `Chunk` is the evidence unit used everywhere.
2. `pipeline.py` — `ask()` shows the five online stages in order.
3. `retrieval/query_planning.py` — rewriting and HyDE create extra search views.
4. `retrieval/search.py` — vector search and reciprocal-rank fusion maximize recall.
5. `retrieval/reranker.py` — the OpenAI reranker improves precision.
6. `generation.py` and `guardrails.py` — generation is constrained and citations
   are checked mechanically.
7. `evaluation/judge.py` — the three evaluation metrics inspect the exact final
   contexts and answer from that run.

Leave ingestion, the HTTP API, Qdrant publication, containers, and deployment
until the core path is understood.

## Follow the data shapes

```text
str question
  -> SanitizedQuestion
  -> list[QueryVariant]
  -> list[Candidate]            broad recall set
  -> list[tuple[Chunk, score]]  final answer contexts
  -> generated JSON            answer + requested citations
  -> Answer                    checked public result
```

The most important vocabulary distinction is `candidates` versus
`final_contexts`. Candidates are allowed to be broad and noisy. Final contexts
are the small evidence set that generation, citation checks, and evaluation all
use.

The three candidate scores also have different jobs:

- `best_similarity` is the best vector similarity from any search view.
- `real_query_score` ignores HyDE and protects the evidence floor.
- `rrf_score` combines ranks from several query views and is used only for
  ordering.

## A 60-minute class

### 1. Baseline retrieval (10 minutes)

Run one question with `query_mode="original"` and reranking disabled. Show that
embedding search is good at recall but can return passages that are only
superficially related.

### 2. Query rewriting and HyDE (15 minutes)

Turn on `rewrite`, then `hyde`, then `hybrid`. Emphasize that these are recall
techniques:

- rewriting expresses the same information need in search-friendly language;
- decomposition creates narrower subqueries for multi-part questions;
- HyDE writes a hypothetical answer-shaped passage to search with;
- the original question is always retained;
- hypothetical text is never passed to generation as evidence.

### 3. Reranking (10 minutes)

Enable reranking and compare the final contexts. The OpenAI listwise reranker
sees the question and the candidate set together, returns an exact candidate
ordering, and assigns explicit relevance grades. Explain the trade-off: better
precision costs an additional model call and latency.

### 4. Guardrails and citations (10 minutes)

Trace one successful answer and one refusal. Focus on layered checks:

- input checks run before model or database calls;
- the evidence floor prevents “best of an irrelevant index” answers;
- generation receives only final real chunks;
- citation IDs must be selected chunks and quotes must occur in their text;
- provider failures become safe, redacted errors.

These deterministic checks are useful engineering controls, not a complete
safety or moderation system.

### 5. Evaluation (15 minutes)

Use the same run to explain three separate questions:

- **Context relevance:** did the final contexts help answer the question?
- **Answer relevance:** did the answer directly address the question?
- **Faithfulness:** are the answer's factual claims supported by the contexts?

Do not collapse them into one score. An answer can be relevant but
unfaithful, or faithful to passages that were themselves irrelevant.

## Suggested classroom experiments

1. Disable rewriting and reranking, record the retrieved contexts, then enable
   one stage at a time.
2. Ask a multi-part question and inspect the generated subqueries.
3. Force a HyDE-only match and verify that it cannot pass the evidence guard by
   itself.
4. Insert an invented citation ID in a test response and watch the citation
   validator remove it.
5. Change one golden question and compare deterministic checks with LLM-judge
   scores.
6. Switch from FAISS to Qdrant and verify that `pipeline.py` does not change.

## What students should be able to explain

By the end, students should be able to say why recall and precision are
separate stages, why HyDE is not evidence, why reranking happens after vector
search, why citations need mechanical validation, why the three evaluation
metrics differ, and why a production database does not fix a weak retrieval
pipeline.
