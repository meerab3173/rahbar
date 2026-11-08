"""Query embedding. The model is the pin (section 4.4): the API must embed with exactly
the model the indexer used, or retrieval silently returns nonsense.

Three families, chosen by EMBED_MODEL:

* any Ollama model name (default ``nomic-embed-text``): POST {OLLAMA_URL}/api/embed
* ``hash-bow-768``: a deterministic bag-of-words hashing embedding with no model and no
  network. CI's e2e-eval job uses it so the whole cluster can be tested offline.
  indexer/pipeline/hashembed.py is the same algorithm; tests pin both to the same output.
* ``dummy-768``: the starter kit's week-1 fake vectors (hash of the whole text).
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import httpx

HASH_MODEL = "hash-bow-768"
DUMMY_MODEL = "dummy-768"
DIM = 768

# nomic-embed-text is trained with task prefixes; the indexer uses "search_document: ".
QUERY_PREFIXES = {"nomic-embed-text": "search_query: "}

_TOKEN = re.compile(r"[a-z]+|\d+(?:[.,]\d+)*")
_STOP = frozenset(
    "a an and are as at be by can do does for from has have how i if in is it its me my "
    "of on or s so than that the their them there this to was we what when where which who "
    "will with would you your".split()
)


class Embedder(Protocol):
    model: str

    def embed(self, text: str) -> list[float]: ...


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def hash_bow(text: str, dim: int = DIM) -> list[float]:
    """Signed feature hashing of unigrams and bigrams, log-scaled, L2-normalised."""
    toks = tokens(text)
    feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
    counts: dict[str, int] = {}
    for f in feats:
        counts[f] = counts.get(f, 0) + 1
    vec = [0.0] * dim
    for feat, n in counts.items():
        h = hashlib.sha1(feat.encode()).digest()
        index = int.from_bytes(h[:4], "big") % dim
        sign = 1.0 if h[4] & 1 else -1.0
        vec[index] += sign * (1.0 + math.log(n))
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        vec[0] = 1.0
        return vec
    return [v / norm for v in vec]


def dummy_vector(text: str, dim: int = DIM) -> list[float]:
    """The starter kit's fake_vector, so the week-1 dummy index can be queried."""
    values: list[float] = []
    seed = text.encode()
    while len(values) < dim:
        seed = hashlib.sha256(seed).digest()
        values.extend((b - 127.5) / 127.5 for b in seed)
    values = values[:dim]
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


class HashEmbedder:
    def __init__(self, model: str = HASH_MODEL) -> None:
        self.model = model

    def embed(self, text: str) -> list[float]:
        return dummy_vector(text) if self.model == DUMMY_MODEL else hash_bow(text)


class OllamaEmbedder:
    def __init__(self, model: str, url: str, timeout: float = 30.0) -> None:
        self.model = model
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.prefix = QUERY_PREFIXES.get(model.split(":")[0], "")

    def embed(self, text: str) -> list[float]:
        response = httpx.post(
            f"{self.url}/api/embed",
            json={"model": self.model, "input": [self.prefix + text]},
            timeout=self.timeout,
        )
        response.raise_for_status()
        vectors = response.json().get("embeddings") or []
        if not vectors:
            raise RuntimeError(f"Ollama returned no embedding for model {self.model}")
        return [float(v) for v in vectors[0]]


def get_embedder(model: str, ollama_url: str) -> Embedder:
    if model in (HASH_MODEL, DUMMY_MODEL):
        return HashEmbedder(model)
    return OllamaEmbedder(model, ollama_url)


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
