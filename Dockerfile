FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

RUN python -m pip install --no-cache-dir uv==0.12.5

COPY pyproject.toml uv.lock README.md MANIFEST.in ./
COPY src ./src

RUN uv sync --locked --no-dev --no-editable


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/app/.venv/bin:$PATH \
    RAG_API_HOST=0.0.0.0 \
    RAG_API_PORT=8477 \
    RAG_DATA_DIR=/var/lib/research-rag \
    RAG_CORPUS_PATH=/app/config/papers.yaml

WORKDIR /app

RUN groupadd --system rag && useradd --system --gid rag --home-dir /app rag

COPY --from=builder /app/.venv /app/.venv
COPY config ./config

RUN mkdir -p /var/lib/research-rag \
    && chown -R rag:rag /app /var/lib/research-rag

USER rag
EXPOSE 8477

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8477/healthz', timeout=2)"

CMD ["research-rag", "serve", "--host", "0.0.0.0", "--port", "8477"]
