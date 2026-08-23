"""The local FAISS vector-store implementation.

FAISS is a fast vector-search library, not a network database. This backend is
the zero-infrastructure default for lessons, notebooks, and one-process apps.

A vector store holds vectors, finds nearest neighbours, and persists both the
vectors and their chunk metadata. New builds are written to a staging directory
and become visible with one atomic ``CURRENT`` pointer swap. The loader also
understands the original two-file layout so existing students do not re-embed.
"""

import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from pydantic import BaseModel, ConfigDict

from research_rag.errors import IndexUnavailable
from research_rag.models import Chunk
from research_rag.retrieval.embeddings import embed
from research_rag.settings import Settings

SCHEMA_VERSION = 1
_SAFE_BUILD_ID = re.compile(r"^[a-f0-9]{32}$")


class IndexManifest(BaseModel):
    """Compatibility contract stored beside every new index build."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int
    build_id: str
    embedding_model: str
    vector_dimension: int
    chunk_count: int
    corpus_sha256: str
    chunks_sha256: str
    created_at: str


@dataclass(frozen=True)
class LoadedFaissBuild:
    """Validated files behind the active ``CURRENT`` pointer."""

    index: faiss.Index
    chunks: list[Chunk]
    build_id: str


def build(
    chunks: list[Chunk],
    *,
    corpus_sha256: str | None = None,
    settings: Settings | None = None,
) -> IndexManifest:
    """Embed chunks and atomically publish one self-describing index build."""

    if not chunks:
        raise ValueError("cannot build an empty index")
    if len({chunk.id for chunk in chunks}) != len(chunks):
        raise ValueError("chunk ids must be unique")

    if settings is None:
        raise TypeError("settings are required by a concrete vector store")
    active = settings
    vectors = embed([chunk.text for chunk in chunks], settings=active)
    if (
        vectors.ndim != 2
        or vectors.shape[0] != len(chunks)
        or vectors.shape[1] < 1
        or not np.isfinite(vectors).all()
    ):
        raise ValueError("embedding response has an unexpected shape or value")

    index = faiss.IndexFlatIP(vectors.shape[1])  # unit-vector IP == cosine
    index.add(vectors)
    chunks_bytes = json.dumps(
        [asdict(chunk) for chunk in chunks], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    chunks_sha256 = hashlib.sha256(chunks_bytes).hexdigest()
    build_id = uuid.uuid4().hex
    manifest = IndexManifest(
        schema_version=SCHEMA_VERSION,
        build_id=build_id,
        embedding_model=active.embedding_model,
        vector_dimension=int(vectors.shape[1]),
        chunk_count=len(chunks),
        corpus_sha256=corpus_sha256 or chunks_sha256,
        chunks_sha256=chunks_sha256,
        created_at=datetime.now(UTC).isoformat(),
    )

    index_dir = active.index_dir
    builds_dir = index_dir / "builds"
    builds_dir.mkdir(parents=True, exist_ok=True)
    staged = builds_dir / f".staging-{build_id}"
    final = builds_dir / build_id
    staged.mkdir()
    try:
        faiss.write_index(index, str(staged / "index.faiss"))
        (staged / "chunks.json").write_bytes(chunks_bytes)
        (staged / "manifest.json").write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )
        _load_build(staged, active.embedding_model)
        os.replace(staged, final)
        _atomic_write_text(index_dir / "CURRENT", build_id + "\n")
    except Exception:
        shutil.rmtree(staged, ignore_errors=True)
        raise

    _load_cached.cache_clear()
    return manifest


def load(settings: Settings) -> "FaissVectorStore":
    """Load and validate the active build, cached until the process restarts."""

    loaded = _load_cached(str(settings.index_dir), settings.embedding_model)
    return FaissVectorStore(
        index=loaded.index,
        chunks=loaded.chunks,
        settings=settings,
        build_id=loaded.build_id,
    )


def clear_cache() -> None:
    """Release cached index objects when an application lifecycle ends."""

    _load_cached.cache_clear()


@lru_cache(maxsize=4)
def _load_cached(index_dir_text: str, embedding_model: str) -> LoadedFaissBuild:
    index_dir = Path(index_dir_text)
    current = index_dir / "CURRENT"
    try:
        if current.is_file():
            build_id = current.read_text(encoding="utf-8").strip()
            if not _SAFE_BUILD_ID.fullmatch(build_id):
                raise IndexUnavailable("index CURRENT pointer is invalid")
            index, chunks = _load_build(
                index_dir / "builds" / build_id, embedding_model
            )
            return LoadedFaissBuild(index, chunks, build_id)

        # Version 1 compatibility: no manifest/model check was recorded. Counts
        # and dimensions are still validated before serving.
        legacy_index = index_dir / "index.faiss"
        legacy_chunks = index_dir / "chunks.json"
        if legacy_index.is_file() and legacy_chunks.is_file():
            index, chunks = _load_pair(legacy_index, legacy_chunks)
            return LoadedFaissBuild(index, chunks, "legacy")
    except IndexUnavailable:
        raise
    except Exception as exc:
        raise IndexUnavailable("the active index is corrupt or unreadable") from exc

    raise IndexUnavailable("no index found; run: research-rag ingest")


def _load_build(
    build_dir: Path, embedding_model: str
) -> tuple[faiss.Index, list[Chunk]]:
    try:
        manifest = IndexManifest.model_validate_json(
            (build_dir / "manifest.json").read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise IndexUnavailable("index manifest is missing or invalid") from exc
    if manifest.schema_version != SCHEMA_VERSION:
        raise IndexUnavailable("index schema version is unsupported")
    if manifest.embedding_model != embedding_model:
        raise IndexUnavailable("embedding model changed; rebuild the index")

    chunks_path = build_dir / "chunks.json"
    if hashlib.sha256(chunks_path.read_bytes()).hexdigest() != manifest.chunks_sha256:
        raise IndexUnavailable("chunk metadata checksum does not match the manifest")
    index, chunks = _load_pair(build_dir / "index.faiss", chunks_path)
    if index.d != manifest.vector_dimension or len(chunks) != manifest.chunk_count:
        raise IndexUnavailable("index dimensions do not match the manifest")
    return index, chunks


def _load_pair(index_path: Path, chunks_path: Path) -> tuple[faiss.Index, list[Chunk]]:
    try:
        index = faiss.read_index(str(index_path))
        raw_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        chunks = [Chunk(**raw) for raw in raw_chunks]
    except Exception as exc:
        raise IndexUnavailable("index files are corrupt or unreadable") from exc
    if index.ntotal != len(chunks) or index.d < 1:
        raise IndexUnavailable("FAISS vectors and chunk metadata do not match")
    return index, chunks


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


class FaissVectorStore:
    """Loaded FAISS index with the same tiny API as the Qdrant backend."""

    backend = "faiss"

    def __init__(
        self,
        *,
        index: faiss.Index,
        chunks: list[Chunk],
        settings: Settings,
        build_id: str,
    ) -> None:
        self.index = index
        self.chunks = chunks
        self.settings = settings
        self.build_id = build_id

    def search_many(
        self, questions: list[str], *, k: int
    ) -> list[list[tuple[Chunk, float]]]:
        """Embed every query in one call, then search every row in FAISS."""

        if not questions:
            return []
        queries = embed(questions, settings=self.settings)
        if (
            queries.ndim != 2
            or queries.shape[0] != len(questions)
            or queries.shape[1] != self.index.d
            or not np.isfinite(queries).all()
        ):
            raise IndexUnavailable("query embeddings do not match the FAISS index")
        scores, ids = self.index.search(queries, k)
        return [
            [
                (self.chunks[i], float(score))
                for score, i in zip(row_scores, row_ids, strict=True)
                if i != -1
            ]
            for row_scores, row_ids in zip(scores, ids, strict=True)
        ]

    def readiness(self) -> dict[str, object]:
        return {
            "ready": True,
            "backend": self.backend,
            "chunks": len(self.chunks),
            "dimension": int(self.index.d),
            "build_id": self.build_id,
        }
