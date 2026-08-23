"""Offline half of RAG: corpus YAML -> download -> parse -> chunk -> index.

Runs once (and again whenever the paper list changes). The online half - answering
questions - lives in ``research_rag.pipeline`` and never re-does this work.
"""

import argparse
import hashlib
import os
import re
import tempfile
import urllib.request
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from research_rag.ingestion.chunking import chunk_paper
from research_rag.ingestion.parsing import parse_pdf
from research_rag.retrieval.store import build
from research_rag.settings import Settings, get_settings

MAX_PDF_BYTES = 100 * 1024 * 1024
SAFE_PAPER_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,79}$")


class Paper(BaseModel):
    """One validated corpus entry; its ID is safe to use as a filename."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, max_length=80)
    title: str = Field(min_length=1, max_length=500)
    url: str

    @field_validator("id")
    @classmethod
    def safe_id(cls, value: str) -> str:
        if not SAFE_PAPER_ID.fullmatch(value):
            raise ValueError("paper id may contain lowercase letters, digits, _ and -")
        return value

    @field_validator("url")
    @classmethod
    def https_url(cls, value: str) -> str:
        if not value.startswith("https://"):
            raise ValueError("paper URL must use HTTPS")
        return value


def load_corpus(path: Path) -> tuple[list[Paper], str]:
    """Validate corpus configuration and return its reproducibility digest."""

    raw_bytes = path.read_bytes()
    raw = yaml.safe_load(raw_bytes)
    if not isinstance(raw, list) or not raw:
        raise ValueError("corpus must be a non-empty YAML list")
    papers = [Paper.model_validate(item) for item in raw]
    ids = [paper.id for paper in papers]
    if len(ids) != len(set(ids)):
        raise ValueError("paper ids must be unique")
    return papers, hashlib.sha256(raw_bytes).hexdigest()


def ingest(settings: Settings) -> None:
    """Build the configured corpus into a new atomic index version."""

    papers, corpus_sha256 = load_corpus(settings.corpus_path)
    settings.pdf_dir.mkdir(parents=True, exist_ok=True)

    chunks = []
    for paper in papers:
        pdf = settings.pdf_dir / f"{paper.id}.pdf"
        if not pdf.exists():
            print(f"downloading {paper.id} ...")
            _download_pdf(paper.url, pdf, settings)
        elif not _looks_like_pdf(pdf):
            raise ValueError(f"cached file for {paper.id} is not a valid PDF")
        pages = parse_pdf(str(pdf), settings=settings)
        paper_chunks = chunk_paper(paper.id, paper.title, pages)
        chunks.extend(paper_chunks)
        print(f"{paper.id:22s} {len(pages):3d} pages -> {len(paper_chunks):3d} chunks")

    manifest = build(chunks, corpus_sha256=corpus_sha256, settings=settings)
    target = (
        str(settings.index_dir)
        if settings.vector_backend == "faiss"
        else f"Qdrant alias {settings.qdrant_collection}"
    )
    print(
        f"\nindexed {len(chunks)} chunks from {len(papers)} papers "
        f"-> {target} (build {manifest.build_id[:8]})"
    )


def _download_pdf(url: str, destination: Path, settings: Settings) -> None:
    """Download with bounds; publish only after a complete PDF is present."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(settings.download_retries + 1):
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "research-rag/2.0"}
            )
            total = 0
            with (
                os.fdopen(descriptor, "wb") as output,
                urllib.request.urlopen(
                    request, timeout=settings.download_timeout_seconds
                ) as response,
            ):
                while block := response.read(1024 * 1024):
                    total += len(block)
                    if total > MAX_PDF_BYTES:
                        raise ValueError("PDF exceeds the 100 MiB download limit")
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())
            temporary_path = Path(temporary)
            if not _looks_like_pdf(temporary_path):
                raise ValueError("download did not contain a PDF")
            os.replace(temporary_path, destination)
            return
        except Exception as exc:  # retry bounded network and partial-write failures
            last_error = exc
            Path(temporary).unlink(missing_ok=True)
            if attempt == settings.download_retries:
                break
    raise RuntimeError(f"failed to download {destination.name}") from last_error


def _looks_like_pdf(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(5) == b"%PDF-"
    except OSError:
        return False


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(prog="research-rag ingest", description=__doc__)
    parser.add_argument("--corpus", type=Path, default=settings.corpus_path)
    parser.add_argument("--data-dir", type=Path, default=settings.data_dir)
    args = parser.parse_args(argv)
    active = settings.model_copy(
        update={
            "corpus_path": args.corpus.expanduser().resolve(),
            "data_dir": args.data_dir.expanduser().resolve(),
        }
    )
    ingest(active)


if __name__ == "__main__":
    main()
