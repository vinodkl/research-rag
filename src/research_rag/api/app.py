"""Typed FastAPI boundary with liveness, readiness, and a tiny chat page."""

import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.resources import files

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from research_rag import __version__
from research_rag.api.schemas import AskRequest, AskResponse, HealthResponse
from research_rag.clients import close_clients
from research_rag.observability import configure_logging
from research_rag.pipeline import ask
from research_rag.retrieval import store
from research_rag.settings import get_settings

LOGGER = logging.getLogger("research_rag.api")
SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,80}$")
PAGE = files("research_rag.api.static").joinpath("index.html").read_text()

# Backwards-compatible lesson name; AskRequest is the public schema.
Question = AskRequest


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    application.state.settings = settings
    readiness = store.readiness(settings)
    LOGGER.info("startup index_ready=%s", readiness["ready"])
    yield
    store.clear_cache()
    close_clients()


def create_app() -> FastAPI:
    application = FastAPI(
        title="research-rag",
        version=__version__,
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def request_metadata(request: Request, call_next):
        supplied = request.headers.get("x-request-id", "")
        request_id = (
            supplied if SAFE_REQUEST_ID.fullmatch(supplied) else uuid.uuid4().hex
        )
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1_000
        response.headers["x-request-id"] = request_id
        LOGGER.info(
            "request id=%s method=%s path=%s status=%s duration_ms=%.1f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    @application.post("/ask", response_model=AskResponse)
    def post_ask_route(body: AskRequest) -> dict:
        return post_ask(body)

    @application.get("/healthz", response_model=HealthResponse)
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/readyz")
    def ready(request: Request) -> JSONResponse:
        # Lifespan pins one validated configuration for the running process.
        # The fallback keeps direct TestClient calls useful without startup.
        settings = getattr(request.app.state, "settings", None) or get_settings()
        status = store.readiness(settings)
        return JSONResponse(status, status_code=200 if status["ready"] else 503)

    @application.get("/", response_class=HTMLResponse)
    def home_route() -> str:
        return home()

    return application


def post_ask(body: AskRequest) -> dict:
    """Public adapter kept separate so its no-trace invariant is obvious."""

    answer = ask(body.question)
    if getattr(answer, "unavailable", False):
        raise HTTPException(status_code=503, detail="The RAG service is unavailable.")
    return {
        "answer": answer.answer,
        "citations": answer.citations,
        "refused": answer.refused,
    }


def home() -> str:
    return PAGE


app = create_app()
