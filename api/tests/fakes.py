"""Test doubles: no Chroma, no Redis, no model, no network."""

from __future__ import annotations

import fnmatch
from typing import Any

from app.embedding import hash_bow
from app.providers.llm.base import ChatJSONProvider, ChatReply
from app.schemas import Answer, Citation, Passage


def passage(cid: str, text: str, source: str = "rules-2026.pdf", section: str = "1. Attendance",
            score: float = 0.8, superseded: bool = False) -> Passage:
    return Passage(chunk_id=cid, text=text, source=source, section=section, page=1,
                   score=score, superseded=superseded)


class FakeRetriever:
    name = "fake"

    def __init__(self, passages: list[Passage], gen: str = "rahbar-v1") -> None:
        self.passages = passages
        self.gen = gen
        self.calls: list[dict[str, Any]] = []

    def retrieve(self, question: str, k: int, *, vector: list[float] | None = None,
                 include_superseded: bool = False) -> list[Passage]:
        self.calls.append({"question": question, "k": k, "vector": vector,
                           "include_superseded": include_superseded})
        return [p.model_copy() for p in self.passages[:k]]

    def readiness(self) -> dict[str, str]:
        return {"chroma": "ok", "collection": "ok (fake, 3 chunks)", "embed_model": "ok"}

    def generation(self) -> str:
        return self.gen


class FakeEmbedder:
    model = "hash-bow-768"

    def embed(self, text: str) -> list[float]:
        return hash_bow(text)


class ScriptedProvider(ChatJSONProvider):
    """A provider whose model replies are scripted, to exercise validate/retry/fallback."""

    name = "scripted"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.seen: list[list[dict[str, str]]] = []

    def _chat(self, messages: list[dict[str, str]]) -> ChatReply:
        self.seen.append(messages)
        if not self.replies:
            raise TimeoutError("no more scripted replies")
        return ChatReply(text=self.replies.pop(0), tokens_in=100, tokens_out=20)


class FixedProvider:
    """Returns one fixed Answer and counts calls."""

    name = "fixed"

    def __init__(self, answer: Answer) -> None:
        self.answer = answer
        self.calls = 0

    def generate(self, question: str, passages: list[Passage]) -> Answer:
        self.calls += 1
        return self.answer


def answer_json(text: str = "You need 100% attendance [rules-2026.pdf / 1. Attendance].",
                source: str = "rules-2026.pdf", section: str = "1. Attendance",
                answered: bool = True) -> str:
    return Answer(reasoning="r", answer=text,
                  citations=[Citation(source=source, section=section, page=1)] if answered else [],
                  answered_from_context=answered, confidence="high").model_dump_json()


class FakeRedis:
    def __init__(self, up: bool = True) -> None:
        self.h: dict[str, dict[str, str]] = {}
        self.lists: dict[str, list[str]] = {}
        self.up = up

    def ping(self) -> bool:
        if not self.up:
            raise ConnectionError("redis down")
        return True

    def hgetall(self, key: str) -> dict[str, str]:
        return dict(self.h.get(key, {}))

    def hset(self, key: str, field: str, value: str) -> None:
        self.h.setdefault(key, {})[field] = value

    def hlen(self, key: str) -> int:
        return len(self.h.get(key, {}))

    def expire(self, key: str, seconds: int) -> None:
        pass

    def delete(self, key: str) -> int:
        return int(self.h.pop(key, None) is not None)

    def scan_iter(self, match: str = "*") -> list[str]:
        return [k for k in self.h if fnmatch.fnmatch(k, match)]

    def lpush(self, key: str, value: str) -> None:
        self.lists.setdefault(key, []).insert(0, value)

    def ltrim(self, key: str, start: int, end: int) -> None:
        self.lists[key] = self.lists.get(key, [])[start:end + 1]


class FakeCollection:
    def __init__(self, name: str, metadata: dict, rows: list[tuple[str, str, dict, float]]):
        self.name, self.metadata, self.rows = name, metadata, rows
        self.last_query: dict[str, Any] = {}

    def count(self) -> int:
        return len(self.rows)

    def get(self, ids: list[str], include: list[str]) -> dict:
        return {"ids": [], "metadatas": []}

    def query(self, query_embeddings: list[list[float]], n_results: int, where: Any,
              include: list[str]) -> dict:
        self.last_query = {"n_results": n_results, "where": where}
        rows = [r for r in self.rows if where is None or not r[2].get("superseded")]
        rows = rows[:n_results]
        return {"ids": [[r[0] for r in rows]], "documents": [[r[1] for r in rows]],
                "metadatas": [[r[2] for r in rows]], "distances": [[r[3] for r in rows]]}


class PointerCollection:
    def __init__(self, target: str) -> None:
        self.name, self.metadata, self.target = "rahbar-active", {}, target

    def get(self, ids: list[str], include: list[str]) -> dict:
        return {"ids": ["active"], "metadatas": [{"collection": self.target}]}


class FakeChroma:
    def __init__(self, collections: dict[str, Any], up: bool = True) -> None:
        self.collections, self.up = collections, up

    def heartbeat(self) -> int:
        if not self.up:
            raise ConnectionError("chroma down")
        return 1

    def get_collection(self, name: str) -> Any:
        if name not in self.collections:
            raise ValueError(f"collection {name} does not exist")
        return self.collections[name]
