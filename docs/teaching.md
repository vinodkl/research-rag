# Teaching guide

Teach one question from input to cited answer. Start with FAISS so storage does not
distract from retrieval quality; introduce Qdrant after the algorithm.

## Reading order

1. `models.py`: a `Chunk` is the evidence unit.
2. `pipeline.py`: the complete online sequence.
3. `retrieval/query_planning.py`: rewriting and HyDE for recall.
4. `retrieval/search.py`: vector search and rank fusion.
5. `retrieval/reranker.py`: OpenAI listwise reranking for precision.
6. `generation.py` and `guardrails.py`: grounded output and citation checks.
7. `evaluation/judge.py`: context relevance, answer relevance, faithfulness.

Leave ingestion, Qdrant publication, HTTP, and deployment until after the core path.

## A 60-minute class

- **10 min — baseline:** original-query vector search, no reranking.
- **15 min — recall:** compare rewrite, HyDE, and hybrid modes; HyDE is never evidence.
- **10 min — precision:** enable reranking and compare final contexts.
- **10 min — grounding:** trace an answer, a refusal, and a rejected citation.
- **15 min — evaluation:** score the same run on the three independent metrics.

Run progressive comparisons with:

```bash
research-rag eval --limit 3 --query-mode original --no-rerank
research-rag eval --limit 3 --query-mode hybrid --no-rerank
research-rag eval --limit 3 --query-mode hybrid
```

## Useful experiments

1. Ask a multi-part question and inspect its subqueries.
2. Force a HyDE-only match and verify it cannot pass the evidence floor alone.
3. Insert an invented citation ID in a test response and watch it disappear.
4. Switch FAISS to Qdrant and verify that `pipeline.py` does not change.

Students should leave able to distinguish recall from precision, explain why HyDE
is not evidence, separate the three metrics, and diagnose a weak RAG pipeline.
