# Contributing

Keep the project understandable to someone learning RAG for the first time.

1. Create a Python 3.12 virtual environment and install `.[dev]`.
2. Make one focused change. Keep stage logic in the stage that owns it and keep
   external I/O behind the API, ingestion, retrieval, or evaluation boundary.
3. Add offline tests. Provider calls must use injected fakes in the default test
   suite.
4. Run:

   ```bash
   ruff format .
   ruff check .
   mypy
   pytest
   python -m build
   ```

Never commit API keys, `.env`, PDFs, indexes, reports, caches, or editor files.
Corpus changes belong in `config/papers.yaml` and should explain why the source
is relevant. A model-default or embedding change should include evaluation
evidence and an explicit note about whether re-ingestion is required.
