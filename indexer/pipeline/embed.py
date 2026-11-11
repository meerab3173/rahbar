"""Stage 4: Embed. One vector per chunk, from the pinned model only."""

from __future__ import annotations

import time

import httpx

from .config import IndexerSettings
from .hashembed import DUMMY_MODEL, HASH_MODEL, dummy_vector, hash_bow

BATCH = 32
# nomic-embed-text is trained with task prefixes; the API uses "search_query: ".
DOCUMENT_PREFIXES = {"nomic-embed-text": "search_document: "}


def _ollama_batch(texts: list[str], settings: IndexerSettings, client: httpx.Client) -> list[list[float]]:
    for attempt in (1, 2, 3):
        try:
            response = client.post(f"{settings.ollama_url.rstrip('/')}/api/embed",
                                   json={"model": settings.embed_model, "input": texts})
            response.raise_for_status()
            vectors = response.json().get("embeddings") or []
            if len(vectors) != len(texts):
                raise RuntimeError(f"asked for {len(texts)} embeddings, got {len(vectors)}")
            return [[float(x) for x in v] for v in vectors]
        except (httpx.TransportError, httpx.HTTPStatusError):
            if attempt == 3:
                raise
            time.sleep(2 * attempt)
    raise RuntimeError("unreachable")


def embed(texts: list[str], settings: IndexerSettings) -> list[list[float]]:
    """Embed with settings.embed_model and nothing else.

    With Ollama: POST {ollama_url}/api/embed with {"model": ..., "input": [texts]}
    returns {"embeddings": [[...], ...]}. Batched, 32 texts per request.
    """
    model = settings.embed_model
    if model == HASH_MODEL:
        return [hash_bow(t) for t in texts]
    if model == DUMMY_MODEL:
        return [dummy_vector(t) for t in texts]
    prefix = DOCUMENT_PREFIXES.get(model.split(":")[0], "")
    vectors: list[list[float]] = []
    with httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
        for i in range(0, len(texts), BATCH):
            vectors.extend(_ollama_batch([prefix + t for t in texts[i:i + BATCH]],
                                         settings, client))
    dims = {len(v) for v in vectors}
    if len(dims) > 1:
        raise RuntimeError(f"embedding model returned mixed dimensions: {sorted(dims)}")
    return vectors
