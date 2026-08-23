# research-rag

A readable, production-shaped RAG system over machine-learning papers. It
shows PDF ingestion, retrieval, reranking, grounded generation, guardrails,
and basic evaluation without hiding the important work behind a framework.

OpenAI is the only model provider; there is no Cohere dependency. Embeddings
use `text-embedding-3-large`. Query rewriting, HyDE, listwise reranking, and
generation default to `gpt-5.6-sol`; captions and evaluation default to
`gpt-5.6-luna`. FAISS is the local default and Qdrant is the shared production
option.

> **LoRA is not implemented.** Its paper is only a document in the example
> corpus. This project does not fine-tune or train a model.

## Quick start

```bash
git clone https://github.com/coding-parrot/research-rag
cd research-rag

python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
cp .env.example .env
# Paste OPENAI_API_KEY into .env, then export it:
set -a && source .env && set +a

.venv/bin/research-rag ingest
.venv/bin/research-rag serve
```

Open <http://127.0.0.1:8477>. The main endpoints are `POST /ask`, `GET
/healthz`, `GET /readyz`, and `GET /docs`. Evidence refusals return `200` with
`"refused": true`; infrastructure or generation outages return `503`.

## The question pipeline

```text
question
  -> validate and sanitize
  -> original query + optional rewrites/decomposition/HyDE
  -> batched embeddings and vector search
  -> weighted reciprocal-rank fusion
  -> real-query evidence floor
  -> OpenAI listwise reranking
  -> grounded generation with citations
  -> mechanical answer and citation checks
```

The original question is always searched. Rewriting and HyDE improve recall,
but hypothetical HyDE text is never evidence. The OpenAI reranker sees all
real candidates together, returns an exact ordering, and labels each one
`direct`, `supporting`, or `irrelevant`; local code rejects missing, duplicate,
or invented IDs. If expansion or reranking fails, the pipeline uses the safe
original-query/vector fallback.

For a guided reading order and a one-hour class, see
[`docs/teaching.md`](docs/teaching.md). Architecture and evaluation details are
in [`docs/architecture.md`](docs/architecture.md) and
[`docs/evaluation.md`](docs/evaluation.md).

## Project map

```text
config/                       corpus and golden evaluation questions
src/research_rag/
  api/                        HTTP schemas, health endpoints, and tiny UI
  ingestion/                  download -> parse -> chunk -> index
  retrieval/                  plan -> search/fuse -> rerank
    faiss_store.py            local versioned vector index
    qdrant_store.py           shared vector database adapter
  evaluation/                 transparent judges and report runner
  pipeline.py                 complete online question flow
  generation.py               grounded answer and structured citations
  guardrails.py               input, evidence, and output checks
  settings.py                 validated environment configuration
  clients.py                  OpenAI and Qdrant client lifecycle
tests/                        unit, integration, and production tests
docs/                         teaching, architecture, evaluation, operations
```

Start with `pipeline.py`, then follow one stage at a time. Qdrant publication,
containers, and deployment are intentionally later topics.

## Configuration and vector storage

Copy `.env.example`; the application reads exported environment variables and
never stores keys in code or index artifacts. Common controls are:

| Variable | Default | Meaning |
|---|---:|---|
| `RAG_VECTOR_BACKEND` | `faiss` | local FAISS or shared Qdrant |
| `RAG_QUERY_MODE` | `auto` | `original`, `rewrite`, `hyde`, `hybrid`, or `auto` |
| `RAG_RERANK` | `on` | enable OpenAI listwise reranking |
| `RAG_PER_QUERY_K` | `20` | results per query view |
| `RAG_CANDIDATE_K` | `20` | fused candidates for reranking |
| `RAG_CONTEXT_K` | `6` | final passages for generation |

FAISS is production-quality search software, but it is a library and file
format rather than a network database. Use it for lessons, notebooks, CI, and
one-process applications. Qdrant adds shared storage, payload APIs, snapshots,
and replication for multiple application replicas. Switching backend or
embedding model requires ingestion; no outage silently falls back to another
backend.

Try the pinned local Qdrant service with:

```bash
docker compose up -d qdrant
export RAG_VECTOR_BACKEND=qdrant
export QDRANT_URL=http://127.0.0.1:6333
.venv/bin/research-rag ingest
.venv/bin/research-rag serve
```

This Compose service is not highly available. Read
[`docs/operations.md`](docs/operations.md) before a remote deployment.

## Tests and evaluation

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/mypy

.venv/bin/research-rag eval --skip-judge
.venv/bin/research-rag eval --json data/eval/latest.json
```

Evaluation reports context relevance, answer relevance, and faithfulness using
the exact final contexts from the run. LLM judges are noisy regression signals,
not ground truth.

## Open-source design references

These repositories inspired boundaries and terminology; no code is copied and
none is an additional runtime framework.

- [Meta FAISS](https://github.com/facebookresearch/faiss) — local vector search.
- [Qdrant](https://github.com/qdrant/qdrant) — shared vector storage, aliases,
  snapshots, and distributed operation.
- [Microsoft GraphRAG](https://github.com/microsoft/graphrag) — explicit offline
  indexing and online query workflows; its graph architecture is different.
- [Microsoft Azure Search OpenAI Demo](https://github.com/Azure-Samples/azure-search-openai-demo)
  — visible ingestion, API, evaluation, and deployment boundaries.
- [OpenAI Evals](https://github.com/openai/evals) — data-driven evaluation cases
  and machine-readable run records.
- [NVIDIA NeMo Guardrails](https://github.com/NVIDIA-NeMo/Guardrails) — separate
  input, retrieval, and output rails. This project implements only a small
  deterministic subset.

## Corpus

The corpus combines the
[InterviewReady AI-engineering library](https://github.com/InterviewReady/ai-engineering-resources)
with canonical additions such as GPT-3, Chinchilla, LLaMA, DPO, ReAct, DPR,
ColBERT, Self-RAG, GPTQ, and FlashAttention-2. PDFs are downloaded during
ingestion and never committed. Treat `config/papers.yaml` changes like code:
they change what the system can retrieve and answer.
