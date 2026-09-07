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

`ingest` runs `fetch -> parse -> chunk -> embed -> validate-index`.
Intermediate and final files are written under `data/`; `data/index.json` is the local embedding store and is checked by the validation stage.

Run an individual stage with `npm run fetch`, `parse`, `chunk`, `embed`, or `validate-index`. Offline checks run with `npm test`; they do not call providers.

## Search

Search the local index with a question:

```sh
npm run search -- "How does attention work?"
```

The CLI embeds the question, ranks all stored chunks by cosine similarity, and prints the five highest-scoring matches with their score, paper, section, page, and text.

## Grounded answers

The `search` command is the passage-only baseline. The `ask` command generates an answer using retrieved passages as context:

```sh
npm run ask -- "How does attention work?"
```

The answer model is instructed to use only the retrieved passages and return an answer plus chunk IDs and quotes. `ask` retrieves 10 vector candidates, reranks them down to 5, then generates the answer. Set `OPENAI_GENERATION_MODEL` or `OPENAI_RERANK_MODEL` to override the default `gpt-4o-mini`.

Reranking and guardrails are intentionally still minimal. RAG citations are checked against retrieved chunk IDs and verbatim quotes before they are displayed.

## Phase 1 learning highlights

- **Fetch:** downloads configured paper PDFs into `data/papers/`.
- **Parse:** converts each PDF into page-preserving text and handles figure captions.
- **Chunk:** turns pages into retrieval units, splitting large sections and merging tiny ones.
- **Embed:** converts every chunk into a 3,072-dimensional meaning vector using `text-embedding-3-large`; unchanged text reuses its cached vector.
- **Retrieve:** embeds the question once, compares it with every chunk using cosine similarity, and returns the top `k` chunks, not an answer.
- **Validate index:** checks `data/index.json` for its model, dimensions, and vector values.
- **Ingest:** only orchestrates `fetch -> parse -> chunk -> embed -> validate-index` in order.
- **Current boundary:** search returns ranked passages; `ask` adds a first grounded-generation pass, while reranking, strict citation validation, guardrails, and a server remain future work.
