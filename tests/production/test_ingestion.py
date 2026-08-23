"""Corpus and download boundaries reject unsafe or incomplete input offline."""

from pathlib import Path

import pytest
import yaml

from research_rag.ingestion import service


def _write_yaml(path: Path, value) -> None:
    path.write_text(yaml.safe_dump(value), encoding="utf-8")


def test_corpus_load_returns_validated_papers_and_reproducible_digest(tmp_path: Path):
    corpus = tmp_path / "papers.yaml"
    entries = [
        {
            "id": "attention",
            "title": "Attention Is All You Need",
            "url": "https://example.test/attention.pdf",
        }
    ]
    _write_yaml(corpus, entries)

    papers, first_digest = service.load_corpus(corpus)
    _, second_digest = service.load_corpus(corpus)

    assert papers[0].id == "attention"
    assert len(first_digest) == 64
    assert first_digest == second_digest


@pytest.mark.parametrize(
    "entries",
    [
        [],
        [
            {
                "id": "../escape",
                "title": "Unsafe ID",
                "url": "https://example.test/paper.pdf",
            }
        ],
        [
            {
                "id": "paper",
                "title": "Insecure URL",
                "url": "http://example.test/paper.pdf",
            }
        ],
        [
            {
                "id": "paper",
                "title": "Unknown field",
                "url": "https://example.test/paper.pdf",
                "surprise": True,
            }
        ],
        [
            {
                "id": "duplicate",
                "title": "First",
                "url": "https://example.test/first.pdf",
            },
            {
                "id": "duplicate",
                "title": "Second",
                "url": "https://example.test/second.pdf",
            },
        ],
    ],
)
def test_corpus_validation_rejects_unsafe_or_ambiguous_entries(tmp_path: Path, entries):
    corpus = tmp_path / "papers.yaml"
    _write_yaml(corpus, entries)

    with pytest.raises(ValueError):
        service.load_corpus(corpus)


def test_failed_download_removes_partial_file_and_temporary_file(
    monkeypatch, tmp_path: Path, settings_factory
):
    class InterruptedResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size: int) -> bytes:
            if not hasattr(self, "started"):
                self.started = True
                return b"%PDF-incomplete"
            raise ConnectionResetError("offline simulated interruption")

    monkeypatch.setattr(
        service.urllib.request,
        "urlopen",
        lambda _request, timeout: InterruptedResponse(),
    )
    destination = tmp_path / "paper.pdf"
    settings = settings_factory(download_retries=0)

    with pytest.raises(RuntimeError, match="failed to download"):
        service._download_pdf("https://example.test/paper.pdf", destination, settings)

    assert not destination.exists()
    assert list(tmp_path.glob(".paper.pdf.*")) == []
