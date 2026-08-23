"""The HTTP boundary exposes stable schemas without leaking internal traces."""

from importlib import import_module
from types import SimpleNamespace

from fastapi.testclient import TestClient

api = import_module("research_rag.api.app")


def test_liveness_and_unready_status_are_distinct(monkeypatch):
    monkeypatch.setattr(
        api.store,
        "readiness",
        lambda _settings=None: {
            "ready": False,
            "backend": "qdrant",
            "error": "IndexUnavailable",
        },
    )
    client = TestClient(api.create_app())

    live = client.get("/healthz")
    ready = client.get("/readyz")

    assert live.status_code == 200
    assert live.json() == {"status": "ok"}
    assert ready.status_code == 503
    assert ready.json() == {
        "ready": False,
        "backend": "qdrant",
        "error": "IndexUnavailable",
    }


def test_ready_endpoint_reports_only_non_sensitive_index_metadata(monkeypatch):
    monkeypatch.setattr(
        api.store,
        "readiness",
        lambda _settings=None: {
            "ready": True,
            "backend": "qdrant",
            "status": "green",
            "chunks": 12,
            "dimension": 3,
            "build_id": "a" * 32,
            "collection": "research-rag",
        },
    )

    response = TestClient(api.create_app()).get("/readyz")

    assert response.status_code == 200
    assert response.json() == {
        "ready": True,
        "backend": "qdrant",
        "status": "green",
        "chunks": 12,
        "dimension": 3,
        "build_id": "a" * 32,
        "collection": "research-rag",
    }
    assert "url" not in response.json()
    assert "api_key" not in response.json()


def test_ask_response_omits_private_pipeline_trace(monkeypatch):
    private_trace = {"hypothetical_document": "must stay private"}
    monkeypatch.setattr(
        api,
        "ask",
        lambda _question: SimpleNamespace(
            answer="The indexed paper supports the answer.",
            citations=[
                {
                    "chunk_id": "paper:0",
                    "quote": "paper supports the answer",
                    "source": "A Test Paper, 2 Method, p.3",
                }
            ],
            refused=False,
            trace=private_trace,
        ),
    )

    response = TestClient(api.create_app()).post(
        "/ask", json={"question": "What does the paper support?"}
    )

    assert response.status_code == 200
    assert set(response.json()) == {"answer", "citations", "refused"}
    assert "trace" not in response.text
    assert "hypothetical_document" not in response.text


def test_ask_request_rejects_short_and_unknown_fields():
    client = TestClient(api.create_app())

    assert client.post("/ask", json={"question": "Hi"}).status_code == 422
    assert (
        client.post(
            "/ask", json={"question": "A valid question?", "debug": True}
        ).status_code
        == 422
    )


def test_operational_failure_is_503_not_a_content_refusal(monkeypatch):
    monkeypatch.setattr(
        api,
        "ask",
        lambda _question: SimpleNamespace(
            answer="redacted infrastructure failure",
            citations=[],
            refused=True,
            unavailable=True,
        ),
    )

    response = TestClient(api.create_app()).post(
        "/ask", json={"question": "What does the paper say?"}
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "The RAG service is unavailable."}
