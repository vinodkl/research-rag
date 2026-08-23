"""Shared domain data that crosses ingestion, retrieval, and generation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """The atomic unit stored, retrieved, reranked, cited, and evaluated."""

    id: str  # "paper_name:3" = paper id + running number
    paper: str  # stable paper id
    title: str  # paper title, used in public citations
    section: str  # heading this chunk belongs to
    page: int  # 1-based page where the section starts
    text: str
