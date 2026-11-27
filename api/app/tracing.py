"""Per-query trace records (Assignment 03, Part D; RAGOps paper 3.2.3, 5.1, Table 2).

Every POST /api/ask produces ONE JSON trace record:

    trace_id, timestamp, user query, rewritten query (a follow-up is retrieved with the
    previous question), embedding model and version, collection name and generation,
    retrieved chunks with document ids, versions and scores, filters applied, the
    reranked order, a prompt hash, model and parameters, the response, citations,
    guardrail actions, latency per stage, cache hit or miss

It goes to two places (F4):
  * standard output, as one JSON log line from the logger "rahbar.trace", so
    `kubectl logs deploy/api | grep rahbar.trace` shows it;
  * TRACE_DIR (/traces, a PersistentVolumeClaim): traces-<pod>.jsonl gets the record,
    and prompts/<sha256>.json the exact messages sent to the model, so
    `python -m app.tracetool show <trace_id>` can print the exact context and prompt
    that produced an answer (D2) after the pod is gone.

The trace id is the request id (X-Request-ID), so one id ties the HTTP log line, the
trace and the response together.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

log = logging.getLogger("rahbar.trace")

trace_var: ContextVar[dict[str, Any] | None] = ContextVar("trace", default=None)


def now_iso() -> str:
    t = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{int(t * 1000) % 1000:03d}Z"


def start(trace_id: str, question: str) -> dict[str, Any]:
    trace: dict[str, Any] = {
        "trace_id": trace_id, "timestamp": now_iso(), "query": question,
        "rewritten_query": None, "embedding": {}, "collection": {}, "filters": None,
        "retrieved": [], "reranked": [], "prompt": None, "model": {}, "response": None,
        "citations": [], "guardrails": [], "latency_ms": {}, "cache": {"hit": False},
        "outcome": {}, "pod": os.environ.get("HOSTNAME", "local"),
    }
    trace_var.set(trace)
    return trace


def current() -> dict[str, Any] | None:
    return trace_var.get()


def rail(name: str, action: str, **detail: Any) -> None:
    """Record what a guardrail did: pass, flagged, blocked, abstain, dropped, ..."""
    trace = current()
    if trace is not None:
        trace["guardrails"].append({"rail": name, "action": action, **detail})


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Time one pipeline stage into latency_ms[name] (added up if it runs twice)."""
    started = time.perf_counter()
    try:
        yield
    finally:
        ms = round((time.perf_counter() - started) * 1000, 1)
        trace = current()
        if trace is not None:
            trace["latency_ms"][name] = round(trace["latency_ms"].get(name, 0.0) + ms, 1)


def prompt_hash(messages: list[dict[str, str]]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()


def record_prompt(messages: list[dict[str, str]], attempt: int) -> str:
    """Remember the exact messages of one model call; the record keeps only their hash."""
    digest = prompt_hash(messages)
    trace = current()
    if trace is not None:
        prompts = trace.setdefault("_prompts", {})
        prompts[digest] = messages
        if attempt == 1:
            trace["prompt"] = {"sha256": digest, "attempts": [digest]}
        elif trace.get("prompt"):
            trace["prompt"]["attempts"].append(digest)
    return digest


class TraceSink:
    """Writes finished traces to stdout and, when TRACE_DIR is set, to the volume."""

    def __init__(self, directory: str = "", stdout: bool = True) -> None:
        self.dir = Path(directory) if directory else None
        self.stdout = stdout
        pod = os.environ.get("HOSTNAME", "local")
        self.file = self.dir / f"traces-{pod}.jsonl" if self.dir else None

    def write(self, trace: dict[str, Any]) -> None:
        prompts: dict[str, Any] = trace.pop("_prompts", {})
        if self.stdout:
            log.info("trace", extra={"trace": trace})
        if self.dir is None or self.file is None:
            return
        try:
            (self.dir / "prompts").mkdir(parents=True, exist_ok=True)
            for digest, messages in prompts.items():
                path = self.dir / "prompts" / f"{digest}.json"
                if not path.exists():
                    path.write_text(json.dumps({"sha256": digest, "messages": messages}),
                                    encoding="utf-8")
            with self.file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(trace, default=str) + "\n")
        except OSError as exc:   # a full or missing volume must not fail the request
            log.warning("trace not written to the volume", extra={"error": str(exc)})
