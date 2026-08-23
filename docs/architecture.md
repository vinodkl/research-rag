# Architecture

The project has two lifecycles. Ingestion is slow, offline, and writes artifacts.
Question answering is fast, online, and only reads a validated artifact.

```text
                                  OFFLINE
config/papers.yaml
        |
        v
 download -> parse -> chunk -> embed -> vector-store boundary
                                      |              |
                                      v              v
                              FAISS build       Qdrant collection
                              + CURRENT         + stable alias
                                      \              /
                                       \            /
                                  ONLINE search
                                        |
request -> guards -> query plan -> search/fuse -> rerank -> generate -> guards
                                                                     |
                                                                     v
                                                            public answer + cites
```

## Boundaries

- `api/` translates HTTP into the public request/response schemas. It never
  serializes `PipelineTrace`.
- `ingestion/` owns remote PDFs, parsing, captions, chunks, and index creation.
  Nothing here runs in an ordinary `/ask` request.
- `retrieval/` owns embeddings, query planning, the small vector-store
  interface, FAISS and Qdrant adapters, rank fusion, and reranking. Only real
  indexed `Chunk` objects can leave this boundary as evidence.
- `evaluation/` consumes the exact final contexts retained by a pipeline run.
  It does not silently retrieve a second time.
- `pipeline.py` is the composition root for one online question. Reading it top
  to bottom shows the complete algorithm and every fallback.
- `settings.py` is the only place that interprets environment configuration.
  `clients.py` is the only place that constructs shared OpenAI and Qdrant
  clients.

The code intentionally avoids a dependency-injection framework. Optional test
clients and ordinary Python functions are enough at this scale.

## Retrieval invariants

1. The original, sanitized question is always searched.
2. Rewrites and HyDE may improve recall but are never evidence.
3. HyDE-only similarity cannot satisfy the deterministic evidence floor.
4. Reciprocal-rank-fusion scores order lists; they are not cosine scores.
5. The reranker must return each supplied chunk ID exactly once.
6. Generation sees only the final selected real chunks.
7. Citation quotes must occur verbatim in the chunk they identify.
8. A configured storage backend either succeeds or reports unavailable; the
   application never hides an outage by searching a different backend.

## Vector-store contract

The online pipeline depends on two operations: batch search and readiness. The
ingestion pipeline depends on one build operation. Backend selection happens in
one factory from `RAG_VECTOR_BACKEND`, so no query-planning, reranking,
generation, guardrail, or evaluation code contains a FAISS/Qdrant branch.

Both adapters store the full `Chunk` payload and use cosine similarity with the
same OpenAI embedding model. A backend switch therefore changes persistence and
operations, not the meaning of retrieval. It still requires a fresh ingestion
because a FAISS file cannot become a Qdrant collection by configuration alone.

### FAISS artifact contract

Each new build contains:

```text
data/index/
  CURRENT
  builds/<build-id>/
    index.faiss
    chunks.json
    manifest.json
```

The manifest records schema version, embedding model, vector dimension, chunk
count, corpus digest, chunk checksum, build ID, and creation time. Ingestion
writes and validates a staging build, renames it into place, then atomically
replaces `CURRENT`. A failed build leaves the previously active pointer intact.

The loader retains compatibility with the original `data/index/index.faiss` and
`chunks.json` pair. That layout lacks an embedding-model manifest, so the next
planned ingestion should upgrade it.

### Qdrant publication contract

`QDRANT_COLLECTION` names a stable alias, not the physical collection written
by ingestion. A build named like `research-rag__<build-id>` contains vectors,
chunk payloads, and collection metadata recording schema version, embedding
model, dimension, chunk count, corpus digest, and build ID.

Ingestion follows this sequence:

1. Create a new physical collection with cosine distance.
2. Upload every vector and its real chunk payload.
3. Validate metadata, exact point count, a sample payload, and collection state.
4. Atomically move the stable alias to the validated collection.

Until step 4, the alias continues to select the previous collection. A serving
process resolves that alias once, validates the result, and pins the physical
collection for its lifetime. Rolling restarts therefore move replicas to the
new build without changing one underneath an in-flight request. Old physical
collections are intentionally retained for those replicas and for rollback.

## Failure behavior

- Query-planning failure: search only the original question.
- Expanded batch-search failure: retry only the original question.
- Reranking failure: use guarded vector/fusion order.
- Missing or weak evidence: refuse instead of generating.
- Invalid generation or citations: refuse instead of returning unchecked data.
- Missing/corrupt/incompatible index: readiness fails and the pipeline returns a
  redacted operational refusal.
- Configured Qdrant unavailable: readiness fails; FAISS is not used as an
  implicit substitute.

Fallback reasons in traces and reports contain exception types, never provider
messages, questions, passages, answers, PII, or keys.
