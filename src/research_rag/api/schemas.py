"""The public HTTP contract, deliberately separate from private traces."""

from pydantic import BaseModel, ConfigDict, Field


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3, max_length=500)


class CitationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    quote: str
    source: str


class AskResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    citations: list[CitationResponse]
    refused: bool


class HealthResponse(BaseModel):
    status: str
