"""Generation uses the settings snapshot passed by the request pipeline."""

from types import SimpleNamespace

from research_rag.generation import generate
from research_rag.models import Chunk
from research_rag.settings import Settings


class FakeCompletions:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content='{"answer": "Grounded.", "citations": []}')
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions())


def test_request_settings_select_the_model_without_rereading_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_GENERATION_MODEL", "gpt-request-generation")
    settings = Settings.from_env()
    monkeypatch.setenv("OPENAI_GENERATION_MODEL", "gpt-later-environment")
    client = FakeClient()
    context = Chunk(
        id="tiny:0",
        paper="tiny",
        title="Tiny Paper",
        section="2 Method",
        page=2,
        text="A grounded passage.",
    )

    result = generate(
        "What is grounded?", [(context, 0.8)], client=client, settings=settings
    )

    assert result == {"answer": "Grounded.", "citations": []}
    call = client.chat.completions.calls[0]
    assert call["model"] == "gpt-request-generation"
    assert "A grounded passage." in call["messages"][1]["content"]
