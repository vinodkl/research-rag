# research-rag

A readable, production-shaped RAG system over machine-learning papers. It shows
students the complete path from PDF ingestion to retrieval, reranking, grounded
generation, guardrails, and basic evaluation without hiding the important work
behind a framework.

Only one model-provider key is needed: OpenAI. There is no Cohere dependency.
An optional remote Qdrant deployment may require its own database API key.
FAISS remains the zero-infrastructure default; Qdrant is the shared,
server-backed option for a multi-replica deployment.

> **LoRA is not part of the implementation.** The LoRA paper is merely one
> document in the example corpus and one question in the evaluation set. This
> project does not fine-tune, adapt, or train a model.

## Quick start

```bash
git clone https://github.com/coding-parrot/research-rag
cd research-rag

python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
cp .env.example .env
# Paste your OPENAI_API_KEY into .env, then export it:
set -a && source .env && set +a

.venv/bin/research-rag ingest
.venv/bin/research-rag serve
```

Open <http://127.0.0.1:8477>. Useful endpoints:

- `POST /ask` — `{"question": "How does HyDE improve retrieval?"}`
- `GET /healthz` — process liveness
- `GET /readyz` — index readiness and non-sensitive index metadata
- `GET /docs` — generated API documentation

Expected safety/evidence refusals are normal `200` responses with
`"refused": true`; an index or generation outage is an HTTP `503`.

When `RAG_VECTOR_BACKEND=faiss` (the default), the existing pre-upgrade index
still loads. The next ingestion creates a versioned build with a manifest and
atomically switches the `CURRENT` pointer.

## What happens to a question

```text
question
  -> normalize, validate, and redact common PII
  -> original query + optional rewrite/decomposition/HyDE plan
  -> batched OpenAI embeddings + configured vector-store retrieval
  -> weighted reciprocal-rank fusion
  -> evidence floor based on the real question
  -> OpenAI listwise reranking
  -> grounded answer with structured citations
  -> mechanical output and citation checks
```

The original question is always searched. HyDE creates a hypothetical passage
only to improve search; that passage can never enter the answer context or count
as evidence.

### Reranker and query rewriter

The quality-first default for both stages is `gpt-5.6-sol` through the OpenAI
Responses API.

- **Query planner:** one structured call produces up to three search-friendly
  rewrites or atomic subqueries plus one HyDE passage. One call keeps the lesson
  and latency budget understandable.
- **Reranker:** one listwise call sees the original question and every candidate,
  returns an exact permutation, and labels each chunk `direct`, `supporting`, or
  `irrelevant`. Local code rejects missing, duplicate, or invented IDs.

These are LLM-based quality stages, not a Cohere endpoint and not a fine-tuned
cross-encoder. If either optional stage fails, retrieval safely falls back to the
original query or guarded vector order. Generation also defaults to
`gpt-5.6-sol`; figure captions and offline judges default to `gpt-5.6-luna`;
embeddings use `text-embedding-3-large`.

## Project map

```text
config/
  papers.yaml                 editable corpus manifest
  golden_questions.yaml      visible evaluation regression set

src/research_rag/
  api/                        typed HTTP boundary and tiny static UI
  ingestion/                  download -> parse -> chunk -> index
  retrieval/                  embed -> plan -> FAISS/Qdrant -> fuse -> rerank
  evaluation/                 transparent judges and report runner
  pipeline.py                 online stages in one readable sequence
  generation.py               grounded answer + structured citations
  guardrails.py               deterministic input/evidence/output checks
  settings.py                 validated environment configuration
  clients.py                  shared OpenAI and Qdrant client lifecycle
  cli.py                      serve / ingest / eval command dispatcher

tests/                        offline unit, integration, and production tests
docs/                         architecture, evaluation, and operations notes
.github/workflows/ci.yml      lint, type, test, build, clean-wheel smoke
Dockerfile                    non-root production image
compose.yaml                  pinned one-node Qdrant environment for local work
uv.lock                       reproducible runtime and CI dependency graph
```

Start with [`pipeline.py`](src/research_rag/pipeline.py), then open one stage at
a time. [`docs/architecture.md`](docs/architecture.md) explains why each boundary
exists.

## Configuration

Copy `.env.example`, paste the key later, and export it into the process. The
application never opens `.env` itself and never stores a key in code or index
artifacts.

Common controls:

| Variable | Default | Meaning |
|---|---:|---|
| `RAG_VECTOR_BACKEND` | `faiss` | `faiss` for local files or `qdrant` for a shared service |
| `RAG_QUERY_MODE` | `auto` | `original`, `rewrite`, `hyde`, `hybrid`, or conditional `auto` |
| `RAG_RERANK` | `on` | enable OpenAI listwise reranking |
| `RAG_PER_QUERY_K` | `20` | vector results per query view |
| `RAG_CANDIDATE_K` | `20` | fused candidates sent to reranking |
| `RAG_CONTEXT_K` | `6` | final passages sent to generation |
| `RAG_DATA_DIR` | `./data` | writable PDFs, cache, indexes, and reports |
| `RAG_CORPUS_PATH` | `./config/papers.yaml` | corpus manifest |

All model names, database settings, API timeouts/retries, server settings, and
ingestion timeouts are documented in [`.env.example`](.env.example). Changing
the embedding model or vector backend requires a new ingestion; each backend
stores and enforces its own embedding contract.

### FAISS or Qdrant?

FAISS is production-quality nearest-neighbour software, but it is a library and
file format—not a network database. That makes it excellent for this lesson,
notebooks, CI, and a single serving process. Keep the default while learning the
pipeline.

Qdrant adds a shared service, payload storage, operational APIs, snapshots, and
cluster replication. Choose it when multiple application replicas must query
the same live index. The query planner, HyDE, fusion, OpenAI reranker,
generation, guardrails, and evaluation code are identical for both backends;
the application never silently falls back from one to the other.

To try the pinned one-node Qdrant service locally:

```bash
docker compose up -d qdrant
export RAG_VECTOR_BACKEND=qdrant
export QDRANT_URL=http://127.0.0.1:6333

# Required once when switching from FAISS, then for each corpus/model update.
.venv/bin/research-rag ingest
.venv/bin/research-rag serve
```

`compose.yaml` pins `qdrant/qdrant:v1.19.0` for repeatability. It is a local
single-node environment, not a high-availability deployment. See
[`docs/operations.md`](docs/operations.md) before using Qdrant outside a laptop.
The replication and write-consistency factors both default to `1`; set both to
`2` on a suitable multi-node cluster.

## Tests and evaluation

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/mypy

# Deterministic retrieval/citation checks only
.venv/bin/research-rag eval --skip-judge

# Add context relevance, answer relevance, and faithfulness judging
.venv/bin/research-rag eval --json data/eval/latest.json
```

The three judge metrics are intentionally small and inspectable:

- **Context relevance:** mean usefulness of each final real context.
- **Answer relevance:** how directly and completely the answer addresses the
  question, scored separately from grounding.
- **Faithfulness:** supported atomic answer claims divided by all valid atomic
  claims, using only the final contexts.

The JSON report includes a schema version, UTC timestamp, safe model/config
metadata, index readiness, aggregates, per-case scores, and degradation signals.
LLM judges are noisy measurements to trend, not objective truth. See
[`docs/evaluation.md`](docs/evaluation.md).

## Production notes

FAISS builds live under `data/index/builds/<build-id>/` with `index.faiss`,
`chunks.json`, and `manifest.json`. A build is fully written and validated before
an atomic `CURRENT` pointer swap makes it active.

Qdrant ingestion likewise uploads and validates a new physical collection, then
atomically moves the stable `QDRANT_COLLECTION` alias. It retains the previous
collection for explicit rollback; operators own snapshot and retention policy.
Each serving process pins the physical build it validated, so rolling restarts
move replicas to the new alias target without changing a build mid-request.
With either backend, readiness validates count, dimension, schema, and embedding
model before traffic should be accepted.

The supplied container runs as a non-root user. A public deployment still needs
TLS, authentication, rate limiting, request-size limits, and concurrency controls
at the ingress. Do not mistake the teaching UI or deterministic guardrails for a
complete moderation, DLP, or adversarial-security system. Operational details are
in [`docs/operations.md`](docs/operations.md).

## Open-source design references

These projects influenced the organization and terminology; they are references,
not copied code or required dependencies.

- [Meta FAISS](https://github.com/facebookresearch/faiss) — the actual local
  vector-search engine and the reference point for exact versus approximate
  nearest-neighbour indexes.
- [Qdrant](https://github.com/qdrant/qdrant) — the popular open-source vector
  database used by the optional server-backed adapter; its collection aliases,
  snapshots, and distributed deployment model inform the production path.
- [Microsoft GraphRAG](https://github.com/microsoft/graphrag) — inspiration for
  separating offline indexing artifacts from online query work and making both
  explicitly configurable. It is a different graph-based architecture.
- [Microsoft Azure Search OpenAI Demo](https://github.com/Azure-Samples/azure-search-openai-demo)
  — inspiration for visible ingestion, API, UI, evaluation, and deployment
  boundaries. Its own documentation also calls out additional production
  hardening, so it is not treated as a drop-in blueprint.
- [OpenAI Evals](https://github.com/openai/evals) — inspiration for data-driven
  evaluation cases, reusable graders, and machine-readable run records.
- [NVIDIA NeMo Guardrails](https://github.com/NVIDIA-NeMo/Guardrails) — inspiration
  for thinking about input, retrieval, and output rails as separate layers. This
  repository implements only a small deterministic subset.

## Corpus

The corpus combines the
[InterviewReady AI-engineering library](https://github.com/InterviewReady/ai-engineering-resources)
with canonical additions such as GPT-3, Chinchilla, LLaMA, DPO, ReAct, DPR,
ColBERT, Self-RAG, GPTQ, and FlashAttention-2. PDFs are downloaded during
ingestion and are never committed. Review changes to `config/papers.yaml` like
code because they change what the system can retrieve and answer.
