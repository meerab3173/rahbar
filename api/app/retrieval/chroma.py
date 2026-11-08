"""The real retriever (sections 3.2 and 4.4). It reads Chroma; it never writes to it.

Which collection: the indexer builds every index into a new, versioned collection
(``rahbar-v<epoch>``) and only when it is complete flips a one-record pointer collection
(``rahbar-active``) to it (ADR 0001). This class follows the pointer, so a re-index is
picked up without a restart and the API never reads a half-built index. If no pointer
exists (the starter kit's ``--dummy`` index), it reads CHROMA_COLLECTION directly.

The pin: the collection's metadata records the embedding model and dimension the indexer
used. A mismatch with this process's EMBED_MODEL makes readiness fail and is logged with
both values, instead of silently serving garbage.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any

from ..config import Settings
from ..embedding import Embedder, get_embedder
from ..schemas import Passage

log = logging.getLogger("rahbar.retrieval")

POINTER_SUFFIX = "-active"
POINTER_ID = "active"
_RESOLVE_SECONDS = 10.0

# Exact terms embeddings blur (section 2, property 7): clause numbers, thresholds,
# course codes, percentages, amounts.
_EXACT = re.compile(
    r"\b[A-Z]{2,4}[- ]?\d{3,4}\b"          # course codes: CS1002, EE-2003
    r"|\b\d+\.\d+(?:\.\d+)?\b"             # clause numbers and decimals: 1.21, 2.00, 6.1
    r"|\b\d+(?:,\d{3})+\b"                 # amounts: 12,000
    r"|\b\d+\s?%"                          # percentages
)


class EmbeddingPinError(RuntimeError):
    """The index was built with a different embedding model than this process uses."""


def exact_terms(text: str) -> list[str]:
    return [t.replace(" ", "").lower() for t in _EXACT.findall(text)]


def exact_boost(question: str, text: str) -> float:
    """Fraction of the question's exact terms that appear verbatim in the chunk."""
    terms = set(exact_terms(question))
    if not terms:
        return 0.0
    haystack = text.replace(" ", "").lower()
    return sum(1 for t in terms if t in haystack) / len(terms)


class ChromaRetriever:
    name = "chroma"

    def __init__(self, settings: Settings, client: Any = None,
                 embedder: Embedder | None = None) -> None:
        self.settings = settings
        # Connected lazily: chromadb.HttpClient() calls the server in its constructor, and
        # the api must start (and stay alive for liveness) even while Chroma is down.
        self._client = client
        self.embedder = embedder or get_embedder(settings.embed_model, settings.ollama_url)
        self._lock = threading.Lock()
        self._active: tuple[float, Any] | None = None
        self._last_conflict: str | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            import chromadb  # the thin client, from chromadb-client in requirements.txt
            from chromadb.config import Settings as ChromaSettings

            self._client = chromadb.HttpClient(
                host=self.settings.chroma_host, port=self.settings.chroma_port,
                settings=ChromaSettings(anonymized_telemetry=False))
        return self._client

    # --- which collection, and is it the one we can read? -------------------------------
    def _resolve(self) -> Any:
        """Return the active collection, re-reading the pointer at most every 10 s."""
        now = time.monotonic()
        with self._lock:
            if self._active and now - self._active[0] < _RESOLVE_SECONDS:
                return self._active[1]
        name = self.settings.collection
        try:
            pointer = self.client.get_collection(self.settings.collection + POINTER_SUFFIX)
            got = pointer.get(ids=[POINTER_ID], include=["metadatas"])
            metas = got.get("metadatas") or []
            if metas and metas[0] and metas[0].get("collection"):
                name = str(metas[0]["collection"])
        except Exception:  # no pointer yet: the dummy index, or a first run in progress
            pass
        collection = self.client.get_collection(name)
        with self._lock:
            self._active = (now, collection)
        return collection

    def pin_conflict(self, collection: Any) -> str | None:
        """None when the index's embedding model matches ours, else a description."""
        meta = collection.metadata or {}
        indexed = meta.get("embed_model")
        if indexed is None:
            return (f"collection {collection.name!r} records no embed_model; "
                    f"rebuild it with the indexer (API EMBED_MODEL={self.settings.embed_model!r})")
        if indexed != self.settings.embed_model:
            return (f"embedding model mismatch: index {collection.name!r} was built with "
                    f"{indexed!r} (dimension {meta.get('dimension')}), but this API has "
                    f"EMBED_MODEL={self.settings.embed_model!r}")
        return None

    def _check_pin(self, collection: Any) -> None:
        conflict = self.pin_conflict(collection)
        if conflict and conflict != self._last_conflict:
            log.error("embedding pin check failed", extra={
                "conflict": conflict, "collection": collection.name,
                "index_embed_model": (collection.metadata or {}).get("embed_model"),
                "api_embed_model": self.settings.embed_model})
        self._last_conflict = conflict
        if conflict:
            raise EmbeddingPinError(conflict)

    def verify_pin_at_boot(self) -> None:
        """Called once when the API boots: log the pin result immediately."""
        try:
            self._check_pin(self._resolve())
            log.info("embedding pin ok", extra={"embed_model": self.settings.embed_model})
        except EmbeddingPinError:
            pass  # already logged; readiness will keep failing
        except Exception as exc:
            log.warning("chroma not reachable at startup", extra={"error": str(exc)})

    def readiness(self) -> dict[str, str]:
        checks: dict[str, str] = {}
        try:
            self.client.heartbeat()
            checks["chroma"] = "ok"
        except Exception as exc:
            checks["chroma"] = f"unreachable: {type(exc).__name__}"
            return checks
        try:
            collection = self._resolve()
        except Exception:
            checks["collection"] = "missing: run the indexer Job"
            return checks
        count = int(collection.count())
        checks["collection"] = f"ok ({collection.name}, {count} chunks)" if count > 0 \
            else f"empty: {collection.name} has 0 chunks"
        try:
            self._check_pin(collection)
            checks["embed_model"] = "ok"
        except EmbeddingPinError as exc:
            checks["embed_model"] = str(exc)
        return checks

    def generation(self) -> str:
        try:
            return str(self._resolve().name)
        except Exception:
            return "unknown"

    # --- the query ----------------------------------------------------------------------
    def retrieve(
        self,
        question: str,
        k: int,
        *,
        vector: list[float] | None = None,
        include_superseded: bool = False,
    ) -> list[Passage]:
        collection = self._resolve()
        self._check_pin(collection)
        if vector is None:
            vector = self.embedder.embed(question)
        dimension = (collection.metadata or {}).get("dimension")
        if dimension and int(dimension) != len(vector):
            raise EmbeddingPinError(
                f"dimension mismatch: index has {dimension}, query vector has {len(vector)}")

        n = max(k, k * max(1, self.settings.candidate_multiplier))
        where = None if include_superseded else {"superseded": False}
        result = collection.query(
            query_embeddings=[vector], n_results=n, where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]

        scored: list[tuple[float, Passage]] = []
        for cid, text, meta, dist in zip(ids, docs, metas, dists):
            meta = meta or {}
            similarity = 1.0 - float(dist)          # Chroma returns cosine distance
            passage = Passage(
                chunk_id=str(cid), text=str(text or ""),
                source=str(meta.get("source", "?")), section=str(meta.get("section", "")),
                page=int(meta["page"]) if meta.get("page") is not None else None,
                score=round(similarity, 4),
                doc_version=str(meta.get("doc_version", "")) or None,
                superseded=bool(meta.get("superseded", False)),
            )
            rank = similarity + self.settings.hybrid_weight * exact_boost(
                question, passage.section + " " + passage.text)
            scored.append((rank, passage))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [p for _, p in scored[:k]]
