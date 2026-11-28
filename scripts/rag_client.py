"""Small helpers shared by the Part B scripts.
# ruff: noqa

Questions go through `make ask`, the same path a user's question takes (Ingress -> api),
so the suite measures the deployed pipeline and needs no knowledge of the API's URL.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def ask(question: str, timeout: int = 900) -> tuple[dict, float]:
    """Ask one question through `make ask`. Returns (response json, seconds)."""
    q = question.replace('"', "").replace("'", "").replace("$", "").replace("`", "")
    t0 = time.perf_counter()
    try:
        out = subprocess.run(["make", "-s", "ask", f"Q={q}"], cwd=ROOT, capture_output=True,
                             text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": f"timed out after {timeout}s"}, time.perf_counter() - t0
    seconds = time.perf_counter() - t0
    text = out.stdout
    m = re.search(r"^\{\s*$", text, flags=re.M)
    if not m:
        return {"error": (out.stderr or text)[-400:]}, seconds
    try:
        return json.loads(text[m.start():]), seconds
    except json.JSONDecodeError as exc:
        return {"error": f"bad json: {exc}"}, seconds


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def section_matches(golden: str, retrieved: str) -> bool:
    """Same rule as eval/run_eval.py: an empty golden section matches any chunk."""
    return not golden or norm(golden) in norm(retrieved)


def ollama(url: str, path: str, payload: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def embed(url: str, model: str, texts: list[str]) -> list[list[float]]:
    try:
        return ollama(url, "/api/embed", {"model": model, "input": texts})["embeddings"]
    except Exception:  # older Ollama: one text per call
        return [ollama(url, "/api/embeddings", {"model": model, "prompt": t})["embedding"]
                for t in texts]
