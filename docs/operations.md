# Operations

## Runtime contract

- Python 3.12 or later.
- One exported `OPENAI_API_KEY`; never bake it into an image or commit `.env`.
- A writable `RAG_DATA_DIR` for PDFs, captions, reports, and FAISS indexes.
- A readable `RAG_CORPUS_PATH` for ingestion.
- The same embedding model at ingestion and serving time.
- Either a local FAISS artifact or a reachable Qdrant service, selected with
  `RAG_VECTOR_BACKEND`.

Settings are validated at process entry. API keys use a masked secret type and
logs contain request IDs, paths, status codes, timing, readiness, counts, and
error types only—never request bodies, model prompts, chunks, answers, or SDK
error messages.

## Local commands

```bash
research-rag ingest --corpus config/papers.yaml --data-dir data
research-rag serve --host 127.0.0.1 --port 8477
research-rag eval --skip-judge
```

The server loads and validates its configured backend lazily, then caches that
validated view for the process lifetime. Build a new index separately and
restart serving processes after a FAISS `CURRENT` or Qdrant alias change. This
explicit reload contract avoids mixed versions in one deployment.

An expected input, evidence, or citation refusal returns HTTP 200 with
`refused=true`. A missing index or unusable generation provider response returns
HTTP 503, allowing load balancers and clients to distinguish content policy from
an outage.

## Choosing the vector backend

The application has one small vector-store interface and two implementations.
It never silently switches backend during an outage.

### FAISS: local default

No database process is required:

```bash
export RAG_VECTOR_BACKEND=faiss
research-rag ingest
research-rag serve
```

FAISS is the clearest path for students, CI, notebooks, and a single application
process. Treat `data/index/` as one versioned deployment artifact. A shared
filesystem can distribute it, but it does not add database transactions,
replication, access control, or operational APIs.

### Qdrant: shared service

The repository's Compose file pins `qdrant/qdrant:v1.19.0`. Start its one-node
development service, select the backend, and ingest the corpus once:

```bash
docker compose up -d qdrant

export RAG_VECTOR_BACKEND=qdrant
export QDRANT_URL=http://127.0.0.1:6333
research-rag ingest
research-rag serve
```

Switching from FAISS is not a file migration: the first Qdrant ingestion embeds
and uploads the corpus into a new collection. Later ingestions also create a new
physical collection, validate it, and atomically move the stable
`QDRANT_COLLECTION` alias. The former collection remains available for
rollback. The Compose service is for local learning only; it is one node,
unencrypted on localhost, and not highly available.

For a managed or remote deployment, set these process variables through the
platform's secret/configuration system:

```text
RAG_VECTOR_BACKEND=qdrant
QDRANT_URL=https://qdrant.example.internal
QDRANT_API_KEY=<secret>
QDRANT_COLLECTION=research-rag
QDRANT_TIMEOUT_SECONDS=10
QDRANT_INDEX_TIMEOUT_SECONDS=300
QDRANT_PREFER_GRPC=off
QDRANT_REPLICATION_FACTOR=2
QDRANT_WRITE_CONSISTENCY_FACTOR=2
```

Use HTTPS for every remote connection, keep Qdrant on a private network, and
rotate its API key. The client validates normal server certificates; do not
disable certificate verification or send an API key over plain HTTP. The two
factor values above assume a suitable multi-node cluster; keep their default of
`1` for the repository's single-node Compose service.

## Qdrant publication, rollback, and retention

`QDRANT_COLLECTION` is the stable application alias. Physical collection names
end in `__<build-id>`. Ingestion does not delete older physical collections;
that is deliberate because publication and retention are different operational
decisions.

For each release:

1. Run one ingestion job at a time and wait for its validation and alias swap.
   The code aborts if it observes another job moving the alias, but the scheduler
   should still enforce single-writer ingestion.
2. Restart serving replicas gradually, checking `/readyz` after each restart.
3. Observe retrieval quality, latency, and errors for an agreed rollback window.
4. Keep at least the current and previous known-good physical collections.
5. Snapshot retained collections, then delete only versions outside the policy
   and only after confirming the stable alias and no running replica use them.

To roll back, atomically move the alias to a known-good physical collection with
the Qdrant alias API, then restart serving replicas. For example, after replacing
the placeholder collection name:

```bash
curl --fail-with-body -X POST "$QDRANT_URL/collections/aliases" \
  -H 'Content-Type: application/json' \
  -H "api-key: $QDRANT_API_KEY" \
  --data '{
    "actions": [
      {"delete_alias": {"alias_name": "research-rag"}},
      {"create_alias": {
        "collection_name": "research-rag__KNOWN_GOOD_BUILD_ID",
        "alias_name": "research-rag"
      }}
    ]
  }'
```

The two alias actions are one Qdrant request. Never delete the collection
currently named by the alias. Follow Qdrant's
[snapshot documentation](https://qdrant.tech/documentation/operations/snapshots/)
and test restoration on the exact deployment topology; a backup that has not
been restored is only an assumption.

## Qdrant production baseline

- Use Qdrant Cloud or a multi-node cluster across failure domains. Both factors
  default to `1` for the local single-node service. On a suitable multi-node
  cluster, set `QDRANT_REPLICATION_FACTOR=2` and
  `QDRANT_WRITE_CONSISTENCY_FACTOR=2`; larger values trade write availability
  and latency for additional durability.
- Keep an odd number of consensus participants—normally three or five—and
  monitor node, shard, replication, disk, memory, and collection health.
- Use HTTPS/TLS, API-key or platform authentication, network isolation, and
  least-privilege access from ingestion and serving workloads.
- Schedule and test snapshots. Retain corpus manifests and collection/build
  metadata so an index can also be reproduced when sources remain available.
- Capacity-test vector count, 3,072-dimensional embeddings, payload size,
  ingestion concurrency, query latency, and recovery time before launch.
- Pin and deliberately upgrade both the Qdrant server and Python client. Review
  compatibility notes and restore a snapshot in staging before production
  upgrades.

See Qdrant's official guides for
[distributed deployment](https://qdrant.tech/documentation/operations/distributed_deployment/),
[security](https://qdrant.tech/documentation/operations/security/), and
[monitoring](https://qdrant.tech/documentation/operations/monitoring/).

## Application container

Build:

```bash
docker build -t research-rag:local .
```

Create or update the persistent FAISS index:

```bash
docker run --rm \
  --env OPENAI_API_KEY \
  --env RAG_VECTOR_BACKEND=faiss \
  --mount type=volume,src=research-rag-data,dst=/var/lib/research-rag \
  research-rag:local research-rag ingest
```

Serve it:

```bash
docker run --rm -p 8477:8477 \
  --env OPENAI_API_KEY \
  --env RAG_VECTOR_BACKEND=faiss \
  --mount type=volume,src=research-rag-data,dst=/var/lib/research-rag,readonly \
  research-rag:local
```

The image runs as an unprivileged user. `/healthz` checks the process;
`/readyz` returns 503 until the configured backend is available and compatible.
When connecting this image to Qdrant in another container, use the Compose
service name (for example `http://qdrant:6333`), not container-local
`127.0.0.1`.

## Public deployment checklist

- Terminate TLS at a trusted ingress.
- Add authentication and authorization appropriate to the corpus.
- Enforce rate, concurrency, body-size, and request-time limits at the ingress.
- Supply secrets from a managed secret store.
- Put ingestion and serving in separate jobs with a shared versioned artifact or
  Qdrant service; never rebuild on web startup.
- Use at least two serving replicas before rolling an index/model change.
- Alert on readiness, provider failure rate, latency, refusal rate, reranker
  fallback rate, and evaluation regressions.
- Back up corpus configuration plus FAISS artifacts or Qdrant snapshots. PDFs
  and vectors are reproducible from the reviewed corpus, subject to source
  availability.

The included regex guardrails are a teaching baseline. Regulated or hostile
deployments need dedicated moderation, DLP, abuse monitoring, and adversarial
testing.
