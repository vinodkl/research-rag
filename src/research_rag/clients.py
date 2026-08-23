"""Lifecycle-managed OpenAI and Qdrant clients."""

from functools import lru_cache

from openai import OpenAI
from qdrant_client import QdrantClient

from research_rag.settings import Settings, get_settings

_CLIENTS: list[OpenAI] = []
_QDRANT_CLIENTS: list[QdrantClient] = []


def openai_client(settings: Settings | None = None) -> OpenAI:
    """Return one client per key/timeout/retry configuration."""

    active = settings or get_settings()
    return _cached_openai_client(
        active.require_openai_api_key(),
        active.openai_timeout_seconds,
        active.openai_max_retries,
    )


@lru_cache(maxsize=4)
def _cached_openai_client(api_key: str, timeout: float, max_retries: int) -> OpenAI:
    client = OpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)
    _CLIENTS.append(client)
    return client


def qdrant_client(settings: Settings | None = None) -> QdrantClient:
    """Return one Qdrant connection pool per non-sensitive configuration."""

    active = settings or get_settings()
    api_key = (
        active.qdrant_api_key.get_secret_value()
        if active.qdrant_api_key is not None
        else None
    )
    return _cached_qdrant_client(
        active.qdrant_url,
        api_key,
        active.qdrant_timeout_seconds,
        active.qdrant_prefer_grpc,
    )


@lru_cache(maxsize=4)
def _cached_qdrant_client(
    url: str, api_key: str | None, timeout: int, prefer_grpc: bool
) -> QdrantClient:
    client = QdrantClient(
        url=url,
        api_key=api_key,
        timeout=timeout,
        prefer_grpc=prefer_grpc,
    )
    _QDRANT_CLIENTS.append(client)
    return client


def close_clients() -> None:
    """Close cached HTTP pools during application shutdown."""

    for openai_connection in _CLIENTS:
        openai_connection.close()
    _CLIENTS.clear()
    _cached_openai_client.cache_clear()
    for qdrant_connection in _QDRANT_CLIENTS:
        qdrant_connection.close()
    _QDRANT_CLIENTS.clear()
    _cached_qdrant_client.cache_clear()
