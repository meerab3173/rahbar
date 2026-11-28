"""Stage 1b: Verify (Assignment 03, A1; RAGOps paper 4.2.1-4.2.2).

Runs after extraction and before chunking. Every document gets a pass or fail, with a
reason, for four checks:

  quality       the extraction worked and produced readable text: not empty (fewer than
                MIN_TEXT_CHARS characters), not garbled (more than MAX_GARBLED_RATIO of
                the characters unreadable), not a failed extraction (the parser raised)
  completeness  the required metadata is there (source, title, date or version, page
                count) and the file is not truncated (a PDF without its %%EOF trailer,
                HTML without </html>, text that stops mid-sentence)
  recency       no newer version of the same document (same doc_id) is in the lake; an
                older version fails this check and is ARCHIVED, not quarantined: it stays
                restorable and is excluded from retrieval through metadata
  uniqueness    not an exact duplicate (same SHA-256 as another document) and not a near
                duplicate (fingerprint cosine >= NEAR_DUP_THRESHOLD with a document that
                has a different doc_id)

A document that fails quality, completeness or uniqueness goes to quarantine/ and is not
indexed. The fingerprint is the mean of the embeddings of three windows of the text
(start, middle, end), with the same model as the index; it is cached by SHA-256, so an
unchanged corpus embeds nothing here either.

Deliberately NOT a check: planted instructions ("ignore your instructions..."). Text is
data (section 3.4 of Assignment 02); the query-time guard handles it and writes its
action into the trace (Part D4).
"""

from __future__ import annotations

import math
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import IndexerSettings
from .lake import Entry
from .models import Document

CHECKS = ("quality", "completeness", "recency", "uniqueness")
WINDOW_CHARS = 1500
# Characters a readable English policy document is made of. Anything else (replacement
# characters, mojibake such as "Ã©" or "â€™", control characters, stray symbols from a
# broken font map) counts towards the garbled ratio.
_READABLE = set(string.ascii_letters + string.digits + string.punctuation + " \t\n\r"
                + "’‘“”–—•…©®°±×÷£€§¶·")
_SENTENCE_END = re.compile(r"[.!?:;)\]\"'’”*|`>]\s*$")

EmbedFn = Callable[[list[str]], list[list[float]]]


@dataclass
class InputDoc:
    """A document this run looks at, before any decision."""

    path: Path
    origin: str                    # "sources", "lake" (carried over) or "batch:<name>"
    file: str
    sha256: str
    meta: dict[str, Any]           # title, doc_id, doc_version, date, superseded, url
    move: bool = False             # inbox files are moved; sources and lake files are not
    entry: Entry | None = None     # the catalog record, if this version is already known

    @property
    def doc_id(self) -> str:
        return str(self.meta.get("doc_id") or "")

    @property
    def version(self) -> str:
        return str(self.meta.get("doc_version") or self.meta.get("date") or "")

    @property
    def doc_key(self) -> str:
        return f"{self.doc_id or self.file}@{self.version or 'unversioned'}"


@dataclass
class Verdict:
    input: InputDoc
    document: Document | None
    error: str = ""
    checks: dict[str, dict[str, str]] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    fingerprint: list[float] | None = None
    decision: str = ""             # "active" | "archived" | "quarantine"

    @property
    def failed(self) -> list[str]:
        return [name for name in CHECKS if self.checks.get(name, {}).get("result") == "fail"]

    def set(self, check: str, ok: bool, reason: str) -> None:
        self.checks[check] = {"result": "pass" if ok else "fail", "reason": reason}


# --- quality ------------------------------------------------------------------------------------
def text_of(doc: Document | None) -> str:
    return "\n".join(p.text for p in doc.pages) if doc else ""


def garbled_ratio(text: str) -> float:
    stripped = "".join(text.split())
    if not stripped:
        return 1.0
    bad = sum(1 for ch in stripped if ch not in _READABLE)
    return bad / len(stripped)


def check_quality(v: Verdict, settings: IndexerSettings) -> None:
    text = text_of(v.document)
    chars = len(text.strip())
    ratio = garbled_ratio(text)
    v.stats.update(chars=chars, garbled_ratio=round(ratio, 4),
                   pages=len(v.document.pages) if v.document else 0,
                   empty_pages=sum(1 for p in v.document.pages if not p.text.strip())
                   if v.document else 0)
    if v.error:
        v.set("quality", False, f"failed extraction: {v.error}")
    elif chars < settings.min_text_chars:
        v.set("quality", False, f"empty: {chars} characters extracted "
                                f"(minimum {settings.min_text_chars})")
    elif ratio > settings.max_garbled_ratio:
        v.set("quality", False, f"garbled: {ratio:.1%} of characters unreadable "
                                f"(maximum {settings.max_garbled_ratio:.0%})")
    else:
        v.set("quality", True, f"{chars:,} characters, {ratio:.2%} unreadable")


# --- completeness ---------------------------------------------------------------------------------
def truncation(path: Path, text: str) -> str:
    """A reason the file looks cut short, or "" if it looks whole."""
    suffix = path.suffix.lower()
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return f"unreadable file: {exc}"
    if not raw:
        return "zero-byte file"
    if suffix == ".pdf":
        if not raw.lstrip().startswith(b"%PDF"):
            return "no %PDF header: not a PDF, or corrupt"
        if b"%%EOF" not in raw[-2048:]:
            return f"no %%EOF trailer in the last 2 KB ({len(raw):,} bytes): the file was cut short"
        return ""
    if suffix in (".html", ".htm"):
        return "" if b"</html" in raw[-4096:].lower() else "no closing </html> tag: the page was cut short"
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return ""
    last = lines[-1].strip()
    if last.startswith("#") or last.startswith("|") or _SENTENCE_END.search(last):
        return ""
    return f"text stops mid-sentence: ...{last[-40:]!r}"


def check_completeness(v: Verdict) -> None:
    meta = v.input.meta
    missing = []
    if not v.input.file:
        missing.append("source")
    if not str(meta.get("title") or "").strip():
        missing.append("title")
    if not (str(meta.get("doc_version") or "").strip() or str(meta.get("date") or "").strip()):
        missing.append("date or version")
    if v.document is None or not v.document.pages:
        missing.append("page count")
    problems = []
    if missing:
        problems.append("missing metadata: " + ", ".join(missing))
    cut = truncation(v.input.path, text_of(v.document))
    if cut:
        problems.append(cut)
    if problems:
        v.set("completeness", False, "; ".join(problems))
    else:
        pages = len(v.document.pages) if v.document else 0
        v.set("completeness", True,
              f"title, version {v.input.version}, {pages} page(s); not truncated")


# --- uniqueness -----------------------------------------------------------------------------------
def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def windows(text: str, size: int = WINDOW_CHARS) -> list[str]:
    """Start, middle and end of the text: three samples that stand for the document."""
    text = " ".join(text.split())
    if len(text) <= 3 * size:
        return [text[i:i + size] for i in range(0, len(text), size)] or [""]
    mid = len(text) // 2 - size // 2
    return [text[:size], text[mid:mid + size], text[-size:]]


def fingerprint(text: str, embed_fn: EmbedFn) -> list[float]:
    vectors = embed_fn(windows(text))
    dim = len(vectors[0])
    mean = [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in mean)) or 1.0
    return [x / norm for x in mean]


# --- the whole stage --------------------------------------------------------------------------------
def verify_all(verdicts: list[Verdict], settings: IndexerSettings, embed_fn: EmbedFn,
               fingerprint_cache: dict[str, list[float]],
               known: dict[str, Entry]) -> list[Verdict]:
    """Run the four checks over this run's documents, in order (sources first, then
    documents already in the lake, then new batches), and decide each one's fate."""
    # 1-2. quality and completeness, document by document
    for v in verdicts:
        check_quality(v, settings)
        check_completeness(v)

    # 3. uniqueness, against every document that came before it in this run
    seen_hash: dict[str, str] = {}      # sha256 -> the file that had it first
    seen_key: dict[str, str] = {}       # doc_key -> the file that had it first
    kept: list[Verdict] = []
    cache_key = settings.embed_model
    for v in verdicts:
        reasons = []
        earlier = seen_hash.get(v.input.sha256)
        if earlier:
            reasons.append(f"exact duplicate of {earlier} (same sha256 "
                           f"{v.input.sha256[:12]}...)")
        else:
            other = seen_key.get(v.input.doc_key)
            if other:
                reasons.append(f"same doc_id and version as {other} ({v.input.doc_key}) "
                               "but different bytes")
        if v.checks["quality"]["result"] == "pass":
            key = f"{cache_key}:{v.input.sha256}"
            if key not in fingerprint_cache:
                fingerprint_cache[key] = fingerprint(text_of(v.document), embed_fn)
            v.fingerprint = fingerprint_cache[key]
            best, nearest = 0.0, ""
            for k in kept:
                if k.fingerprint is None or k.input.doc_id == v.input.doc_id:
                    continue            # versions of one document: recency's job
                sim = cosine(v.fingerprint, k.fingerprint)
                if sim > best:
                    best, nearest = sim, k.input.doc_key
            v.stats.update(nearest=nearest, nearest_similarity=round(best, 4))
            if nearest and best >= settings.near_dup_threshold:
                reasons.append(f"near-duplicate of {nearest} (cosine {best:.3f} >= "
                               f"{settings.near_dup_threshold})")
        if reasons:
            v.set("uniqueness", False, "; ".join(reasons))
        else:
            near = v.stats.get("nearest_similarity")
            detail = (f"; nearest other document {v.stats['nearest']} at cosine {near:.3f}"
                      if v.stats.get("nearest") else "")
            v.set("uniqueness", True, "no exact or near duplicate" + detail)
            seen_hash.setdefault(v.input.sha256, v.input.file)
            seen_key.setdefault(v.input.doc_key, v.input.file)
            if v.checks["quality"]["result"] == "pass" and \
                    v.checks["completeness"]["result"] == "pass":
                kept.append(v)

    # 4. recency, among the documents that survived the other three checks
    groups: dict[str, list[Verdict]] = {}
    for v in kept:
        groups.setdefault(v.input.doc_id or v.input.file, []).append(v)
    current: dict[str, Verdict] = {}
    for doc_id, members in groups.items():
        pinned = [m for m in members
                  if (known.get(m.input.doc_key) and known[m.input.doc_key].pinned)]
        current[doc_id] = pinned[0] if pinned else max(members, key=lambda m: m.input.version)
    for v in verdicts:
        doc_id = v.input.doc_id or v.input.file
        winner = current.get(doc_id)
        if winner is None or winner is v:
            if v.input.meta.get("superseded") is True and winner is v and \
                    len(groups.get(doc_id, [])) == 1:
                v.set("recency", False, "marked superseded in its metadata; no newer "
                                        "version is in the lake")
            elif winner is v and len(groups[doc_id]) > 1:
                why = "restored by request (pinned)" if known.get(v.input.doc_key) and \
                    known[v.input.doc_key].pinned else "latest"
                v.set("recency", True, f"{why} of {len(groups[doc_id])} versions of {doc_id}")
            else:
                v.set("recency", True, f"no other version of {doc_id}")
        elif winner.input.version > v.input.version:
            v.set("recency", False, f"newer version {winner.input.doc_key} exists")
        elif winner.input.version < v.input.version:
            v.set("recency", False, f"{winner.input.doc_key} was restored and is pinned "
                                    "as current; this newer version is held back")
        else:   # the same version under another file name (a copy): not a recency problem
            v.set("recency", True, f"no newer version of {doc_id} "
                                   f"(current: {winner.input.file})")

    # the decision
    for v in verdicts:
        if any(v.checks[c]["result"] == "fail" for c in ("quality", "completeness",
                                                         "uniqueness")):
            v.decision = "quarantine"
        elif v.checks["recency"]["result"] == "fail":
            v.decision = "archived"
        else:
            v.decision = "active"
    return verdicts


def report(run_id: str, verdicts: list[Verdict], settings: IndexerSettings,
           placed: dict[str, str]) -> tuple[dict[str, Any], str]:
    """The verification report: one record per document, and a table for kubectl logs."""
    docs: list[dict[str, Any]] = []
    for v in verdicts:
        docs.append({"doc_key": v.input.doc_key, "file": v.input.file,
                     "origin": v.input.origin, "sha256": v.input.sha256,
                     "checks": v.checks, "decision": v.decision, "stats": v.stats,
                     "now_at": placed.get(v.input.doc_key + "|" + v.input.file, "")})
    counts = {d: sum(v.decision == d for v in verdicts)
              for d in ("active", "archived", "quarantine")}
    data = {"run_id": run_id, "documents": docs, "counts": counts, "thresholds": {
        "min_text_chars": settings.min_text_chars,
        "max_garbled_ratio": settings.max_garbled_ratio,
        "near_dup_threshold": settings.near_dup_threshold,
        "fingerprint_model": settings.embed_model}}
    lines = [f"VERIFICATION REPORT  run {run_id}",
             f"thresholds: min_text_chars={settings.min_text_chars}  "
             f"max_garbled_ratio={settings.max_garbled_ratio}  "
             f"near_dup_threshold={settings.near_dup_threshold}  ({settings.embed_model})",
             ""]
    for d in docs:
        lines.append(f"{d['file']}  [{d['origin']}]  ->  {d['decision'].upper()}")
        for name in CHECKS:
            c = d["checks"].get(name, {"result": "?", "reason": ""})
            lines.append(f"    {name:<12} {c['result'].upper():<4}  {c['reason']}")
        if d["now_at"]:
            lines.append(f"    now at       {d['now_at']}")
        lines.append("")
    lines.append(f"{len(docs)} documents: {counts['active']} active, "
                 f"{counts['archived']} archived, {counts['quarantine']} quarantined")
    fails = {name: sum(1 for v in verdicts if v.checks.get(name, {}).get("result") == "fail")
             for name in CHECKS}
    lines.append("failures by check: " + ", ".join(f"{k} {n}" for k, n in fails.items()))
    data["failures_by_check"] = fails
    return data, "\n".join(lines) + "\n"
