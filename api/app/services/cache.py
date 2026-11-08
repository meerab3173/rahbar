"""The semantic cache (section 3.3), in Redis. ADR 0003 explains why Redis and not a
second Chroma collection: the API must never write to Chroma.

Layout, per index generation `g` (the active Chroma collection's name):
    rahbar:cache:<g>   a hash: entry id -> JSON {"q", "v" (base64 float32), "resp"}
A lookup reads the generation's hash (at most CACHE_MAX_ENTRIES entries) and returns
the best entry whose cosine similarity with the question is >= CACHE_THRESHOLD.

Invalidation on re-index: the key contains the generation, so the moment the indexer
flips the active collection, every old entry stops matching; `invalidate_other_generations`
then deletes the old hashes. The TTL is a second line of defence.

Also here: `store_turn`, step 8's turn record (a hash of the question, never the text;
ADR 0004).
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import struct
import time
import uuid
from typing import Any

from ..embedding import cosine
from .guards import exact_signature

log = logging.getLogger("rahbar.cache")

PREFIX = "rahbar:cache:"
TURNS_KEY = "rahbar:turns"
MAX_TURNS = 1000


def pack(vector: list[float]) -> str:
    return base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode()


def unpack(blob: str) -> list[float]:
    raw = base64.b64decode(blob)
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


class SemanticCache:
    def __init__(self, client: Any, threshold: float, ttl_seconds: int, max_entries: int) -> None:
        self.client = client
        self.threshold = threshold
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._generation_seen: str | None = None

    def ping(self) -> str:
        try:
            self.client.ping()
            return "ok"
        except Exception as exc:
            return f"unreachable: {type(exc).__name__}"

    def invalidate_other_generations(self, generation: str) -> int:
        """Delete every cache hash that does not belong to `generation`."""
        if generation == self._generation_seen:
            return 0
        removed = 0
        for key in list(self.client.scan_iter(match=PREFIX + "*")):
            name = key.decode() if isinstance(key, bytes) else str(key)
            if name != PREFIX + generation:
                removed += int(self.client.delete(name))
        if removed:
            log.info("cache invalidated after re-index",
                     extra={"generation": generation, "removed_keys": removed})
        self._generation_seen = generation
        return removed

    def lookup(self, generation: str, vector: list[float],
               question: str = "") -> tuple[dict | None, float]:
        """Return (cached response, similarity) for the best hit, or (None, best score).
        A hit needs similarity >= threshold AND the same exact terms (guards.py)."""
        self.invalidate_other_generations(generation)
        signature = sorted(exact_signature(question))
        best: tuple[float, dict | None] = (0.0, None)
        for raw in self.client.hgetall(PREFIX + generation).values():
            try:
                entry = json.loads(raw)
                if question and entry.get("terms", signature) != signature:
                    continue
                score = cosine(vector, unpack(entry["v"]))
            except (ValueError, KeyError, TypeError):
                continue
            if score > best[0]:
                best = (score, entry)
        score, entry = best
        if entry is not None and score >= self.threshold:
            return entry["resp"], score
        return None, score

    def store(self, generation: str, question: str, vector: list[float], response: dict) -> bool:
        key = PREFIX + generation
        if int(self.client.hlen(key)) >= self.max_entries:
            return False                      # full: never evict silently, let TTL clear it
        entry = {"q": question, "v": pack(vector), "resp": response, "ts": int(time.time()),
                 "terms": sorted(exact_signature(question))}
        self.client.hset(key, uuid.uuid4().hex, json.dumps(entry))
        self.client.expire(key, self.ttl)
        return True

    def store_turn(self, record: dict) -> None:
        self.client.lpush(TURNS_KEY, json.dumps(record))
        self.client.ltrim(TURNS_KEY, 0, MAX_TURNS - 1)
        self.client.expire(TURNS_KEY, 7 * 86400)


def question_digest(question: str) -> str:
    return hashlib.sha256(question.strip().lower().encode()).hexdigest()[:16]


def get_cache(backend: str, redis_url: str, threshold: float, ttl: int,
              max_entries: int) -> SemanticCache | None:
    if backend != "redis":
        return None
    import redis  # the redis client from requirements.txt

    client = redis.Redis.from_url(redis_url, socket_timeout=2.0, socket_connect_timeout=2.0)
    return SemanticCache(client, threshold, ttl, max_entries)
