"""Offline embeddings with no model and no network (copy of api/app/embedding.py).

``hash-bow-768`` is a signed bag-of-words feature hash used by CI's e2e-eval job, so the
whole cluster can be tested without Ollama. ``dummy-768`` is the starter's fake vector.
Both packages' tests assert the same fingerprint, so the two copies cannot drift.
"""

from __future__ import annotations

import hashlib
import math
import re

HASH_MODEL = "hash-bow-768"
DUMMY_MODEL = "dummy-768"
DIM = 768

_TOKEN = re.compile(r"[a-z]+|\d+(?:[.,]\d+)*")
_STOP = frozenset(
    "a an and are as at be by can do does for from has have how i if in is it its me my "
    "of on or s so than that the their them there this to was we what when where which who "
    "will with would you your".split()
)


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
