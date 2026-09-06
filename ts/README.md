# research-rag TypeScript

Learning port of the local ingestion pipeline.

## Setup

```sh
cp .env.example .env
npm install
npm run ingest
```

Required environment variables:

- `OPENAI_API_KEY`, for embeddings
- `DEEPSEEK_API_KEY`, only when a PDF needs generated figure captions

## Stages

`ingest` runs `fetch -> parse -> chunk -> embed -> store`.
Intermediate and final files are written under `data/`; `data/index.json` is the local embedding store and is validated by the store stage.

Run an individual stage with `npm run fetch`, `parse`, `chunk`, `embed`, or `store`. Offline checks run with `npm test`; they do not call providers.

## Search

Search the local index with a question:

```sh
npm run search -- "How does attention work?"
```

The CLI embeds the question, ranks all stored chunks by cosine similarity, and prints the five highest-scoring matches with their score, paper, section, page, and text.

Reranking and answer generation are intentionally not implemented yet.
