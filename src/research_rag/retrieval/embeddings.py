"""Step 3 - Embedding: text becomes a vector (Week 3: "Embedding Models").

An embedding model maps text to a fixed-length vector so that similar meanings
land near each other. "Near" has a precise definition: cosine similarity. OpenAI's
embeddings come back already normalised to length 1. FAISS therefore uses inner
product, while Qdrant is configured explicitly for cosine distance.

The model is a contract: the same model must embed the chunks at index time and
the query at question time. Swap the model without re-indexing and every search
silently returns nonsense.
"""

import numpy as np

from research_rag.clients import openai_client
from research_rag.settings import DEFAULT_EMBEDDING_MODEL, Settings, get_settings

MODEL = DEFAULT_EMBEDDING_MODEL  # 3072 dimensions with the default model
BATCH = 128  # texts per API request


def api_key() -> str:
    """Compatibility helper; production code obtains the key from Settings."""

    return get_settings().require_openai_api_key()


def embed(
    texts: list[str], *, client=None, settings: Settings | None = None
) -> np.ndarray:
    """List of texts -> matrix of unit-length vectors, one row per text."""
    active = settings or get_settings()
    active_client = client or openai_client(active)

    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH):
        response = active_client.embeddings.create(
            model=active.embedding_model, input=texts[start : start + BATCH]
        )
        vectors.extend(item.embedding for item in response.data)
    return np.array(vectors, dtype="float32")
