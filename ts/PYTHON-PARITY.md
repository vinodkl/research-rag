# TS ↔ Python parity status

Remaining TypeScript work compared with the Python reference implementation
(`src/research_rag/`). Status markers: ✅ done · ◑ partial · ❌ missing.

## 1. Ingestion (`ingestion/service.py`, `retrieval/store.py`, `faiss_store.py`)

| Item | TS status |
|---|---|
| Corpus from YAML with validation (safe id, HTTPS, unique) | ❌ hardcoded `papers.ts` array |
| Download retries, 100 MB cap, `%PDF-` check, atomic temp-file write | ❌ single `fetch()`, no limits |
| Cached-PDF validity check on re-run | ❌ only existence check |
| Settings module (env-driven models, dirs, query mode, k-values) | ❌ scattered `process.env` |
| Index manifest: build id, chunk sha256, model contract, atomic publish, versioned rebuilds | ❌ single `index.json`, no manifest |
| Two backends behind one interface (FAISS / Qdrant) | ❌ JSON-file store only (fine for learning) |
| Chunk-id uniqueness check before indexing | ❌ |

## 2. Retrieval (`query_planning.py`, `retrieval/search.py`)

| Item | TS status |
|---|---|
| Multi-query planning: up to 3 rewrites + decomposition + HyDE, modes (`original/rewrite/hyde/hybrid/auto`) | ◑ single rewrite, no HyDE/decomposition/modes |
| One batched embedding call for all query views | ❌ one embedding call per view |
| Per-kind weights (original 1.0 / rewrite 0.8 / hyde 0.7), guard score excludes HyDE | ❌ unweighted, no guard/vector score split |
| Batch-failure fallback to original query with diagnostics | ❌ |
| `should_expand` heuristic (open-ended/multi-part detection) | ❌ |

## 3. Reranking (`reranker.py`)

| Item | TS status |
|---|---|
| Grade scores (direct 1.0 / supporting 0.5 / irrelevant 0.0) used as the hard rule | ◑ ordering only, grades dropped |
| Fallback keeps vector order + applied/error/model diagnostics | ◑ fallback exists, no diagnostics struct |
| Dedupe, permutation validation | ✅ |
| Rerank on/off toggle | ❌ always on in `ask` |

## 4. Generation + guards (`generation.py`, `guardrails.py`, `pipeline.py`)

| Item | TS status |
|---|---|
| Citations: chunk-id + verbatim quote, dedupe, 10–300 chars, refusal when none valid | ✅ |
| Output guard: empty answer / PII-leak in answer | ❌ |
| Full question guards: NFKC, Luhn-validated cards, redaction record, length | ◑ subset |
| Retrieval score floor on real-query cosine (0.10), HyDE excluded | ◑ floor exists, but applied post-fusion |
| Pipeline result object: `Answer{answer, citations, refused, unavailable, trace}` | ❌ linear CLI, no trace |
| Graceful generation-failure refusal | ❌ uncaught generation errors exit the CLI |

## 5. Missing modules entirely

| Python | TS |
|---|---|
| HTTP API + chat UI + `/health`, `/ready` (`api/`) | ❌ |
| `research-rag serve/ingest/evaluate` CLI | ❌ npm scripts only |
| LLM-as-judge evaluation + runner (`evaluation/`) | ❌ |
| Observability, logging config, typed errors, client lifecycle (`observability.py`, `clients.py`, `errors.py`) | ❌ |
| Answer generation from final contexts only + schema `strict` | ◑ |

## Biggest remaining gaps

- HyDE / decomposition / query modes
- Vector-store manifest layer
- Input/output guard completeness (PII in answers)
- Pipeline trace + API/UI
- Evaluation
