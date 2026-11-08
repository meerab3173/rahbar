"""Stage 3: Chunk. Structure-aware, with overlap.

Unit: a token here is a whitespace-separated word (docs/PIPELINE.md). nomic-embed-text's
real tokenizer gives roughly 1.3 tokens per word on this corpus, well inside its window.

Algorithm (the RAG lecture's order: sections, then paragraphs, then sentences):
  1. Walk the cleaned pages line by line, recognising headings (Markdown #, "5. Academic
     Assessment", "XII - Policy on ...", "6.1 Use in Lab Courses", and short Title Case
     lines such as "Attendance Requirements"). Headings form a stack; a chunk's `section`
     is the top level and the deepest one, e.g.
     "1. Medium of Instruction > Attendance Requirements".
  2. Inside a section, rebuild paragraphs from wrapped lines. A numbered clause
     ("1.21 Students are ..."), a list item or a table row always starts a new one.
  3. Pack whole paragraphs into a chunk up to `size` tokens. A paragraph longer than
     `size` is split into sentences; a sentence longer than `size` into word windows.
  4. A new chunk in the same section starts with the last `overlap` tokens of the
     previous one (whole trailing sentences where they fit). Overlap never crosses a
     section boundary: a heading is a real boundary.
  5. Each chunk's text starts with its section heading, so the embedding knows where
     the text sits ("contextual chunk header"); those words count as stored tokens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .models import Chunk, Document

_MD = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_TOP = re.compile(r"^(\d{1,2})\s*[.)]\s+([A-Z(][^\n]{2,90})$")
_ROMAN = re.compile(r"^([IVXL]{1,6})\s*[-–—.:]\s*([A-Z][^\n]{2,90})$")
_SUB = re.compile(r"^(\d{1,2}\.\d{1,2})\s+([A-Z][^\n]{2,80})$")
_CLAUSE = re.compile(r"^\(?(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)\)?[\s.:)]")
_ITEM = re.compile(r"^(\(?[a-z]{1,2}\)|[a-z][.)]|\(?[ivx]{1,4}\)|[-*]|\d{1,2}[.)])\s+")
_SENTENCE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z(\"'0-9])")
# A short line ending in a number, e.g. "A+ 4.00" or "Islamabad 1,250": a table row whose
# column separators were lost in extraction. Kept on its own line, never re-wrapped.
_ROWLIKE = re.compile(r"^\S.{0,45}\s\d[\d.,%]*$")
_SMALL = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or", "the",
          "to", "with", "vs", "&", "-", "/"}

RANK = {"roman": 1, "top": 2, "sub": 3, "plain": 4}


def ntokens(text: str) -> int:
    return len(text.split())


def _title_like(text: str) -> bool:
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    if not words or len(words) > 9 or len(text) > 80:
        return False
    if text.rstrip()[-1] in ".,;:" or re.search(r"\d\s*$", text):
        return False
    if not text[0].isupper():
        return False
    content = [w for w in words if w.lower() not in _SMALL and re.search(r"[A-Za-z]", w)]
    if not content:
        return False
    caps = sum(1 for w in content if w[0].isupper() or w.isupper())
    return caps / len(content) >= 0.75


def heading(line: str) -> tuple[int, str] | None:
    """(rank, title) when the line is a heading, else None. Lower rank = higher level."""
    s = line.strip()
    if not s or s.startswith("|"):
        return None
    m = _MD.match(s)
    if m:
        level = len(m.group(1))
        return (0 if level == 1 else 10 + level), m.group(2).strip()
    if _ROMAN.match(s) and _title_like(_ROMAN.match(s).group(2)):  # type: ignore[union-attr]
        return RANK["roman"], s
    m = _TOP.match(s)
    if m and _title_like(m.group(2)):
        return RANK["top"], s
    m = _SUB.match(s)
    if m and _title_like(m.group(2)) and len(s.split()) <= 9:
        return RANK["sub"], s
    if _title_like(s) and len(s.split()) >= 2:
        return RANK["plain"], s
    return None


@dataclass
class Para:
    text: str
    page: int


@dataclass
class Section:
    label: str
    paras: list[Para] = field(default_factory=list)


def section_label(stack: list[tuple[int, str]]) -> str:
    """The chapter and the leaf: "5. Academic Assessment > Academic Warning/Probation".
    With three or more levels the middle ones are dropped, so the chapter is never lost."""
    visible = [title for rank, title in stack if rank != 0]
    if len(visible) > 2:
        visible = [visible[0], visible[-1]]
    return " > ".join(visible)


def sections(doc: Document) -> list[Section]:
    """Split the document into sections of paragraphs, tracking the page of each."""
    stack: list[tuple[int, str]] = []
    out: list[Section] = [Section(label="")]
    buf: list[str] = []
    buf_page = 1

    def flush() -> None:
        nonlocal buf
        if buf:
            out[-1].paras.append(Para(" ".join(buf).strip(), buf_page))
            buf = []

    for page in doc.pages:
        for raw in page.text.splitlines():
            line = raw.strip()
            if not line:
                flush()
                continue
            h = heading(line)
            if h is not None and stack and h[1].lower() == stack[-1][1].lower():
                h = None    # a repeated heading (e.g. a table's header row) is text
            if h is not None:
                flush()
                rank, title = h
                while stack and stack[-1][0] >= rank:
                    stack.pop()
                stack.append((rank, title))
                out.append(Section(label=section_label(stack)))
                continue
            row = line.startswith("|") or bool(_ROWLIKE.match(line))
            starts_new = bool(_CLAUSE.match(line) or _ITEM.match(line)) or row
            if starts_new or (buf and (buf[-1].startswith("|") or _ROWLIKE.match(buf[-1]))):
                flush()
            if not buf:
                buf_page = page.page
            buf.append(line)
        # a page break inside a paragraph keeps the paragraph (common in PDFs)
    flush()
    title = next((t for r, t in stack if r == 0), "") if stack else ""
    for s in out:
        if not s.label:
            s.label = title or "Introduction"
    return [s for s in out if s.paras]


def _split_long(text: str, size: int) -> list[str]:
    """A paragraph longer than `size`: sentences, then word windows."""
    pieces: list[str] = []
    for sentence in _SENTENCE.split(text):
        words = sentence.split()
        if len(words) <= size:
            pieces.append(sentence)
        else:
            pieces.extend(" ".join(words[i:i + size]) for i in range(0, len(words), size))
    return pieces


def _tail(text: str, overlap: int) -> str:
    """The last `overlap` tokens of a chunk body, as whole sentences when they fit."""
    if overlap <= 0:
        return ""
    sentences = _SENTENCE.split(text)
    tail: list[str] = []
    count = 0
    for sentence in reversed(sentences):
        n = ntokens(sentence)
        if count + n > overlap:
            break
        tail.insert(0, sentence)
        count += n
    if tail:
        return " ".join(tail)
    return " ".join(text.split()[-overlap:])


def _slug(source: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", Path(source).stem.lower()).strip("-")[:60]


def chunk(doc: Document, size: int, overlap: int, strategy: str = "structure") -> list[Chunk]:
    """Split on sections, then paragraphs, then sentences, keeping each chunk under
    `size` tokens, with `overlap` tokens shared between neighbours. Every chunk keeps
    the section heading it sits under, its page, doc_version and superseded."""
    if strategy == "fixed":
        return chunk_fixed(doc, size, overlap)
    if size <= 0:
        raise ValueError("chunk size must be positive")
    overlap = max(0, min(overlap, size // 2))
    chunks: list[Chunk] = []

    def emit(label: str, body: str, page: int) -> None:
        body = body.strip()
        if not body:
            return
        text = f"{label}\n{body}" if label else body
        chunks.append(Chunk(
            id=f"{_slug(doc.source)}-{len(chunks):04d}", text=text, source=doc.source,
            section=label, page=page, doc_version=doc.doc_version,
            superseded=doc.superseded))

    for sec in sections(doc):
        units: list[Para] = []
        for para in sec.paras:
            if ntokens(para.text) <= size:
                units.append(para)
            else:
                units.extend(Para(piece, para.page) for piece in _split_long(para.text, size))
        current: list[str] = []
        current_tokens = 0
        page = units[0].page if units else 1
        for unit in units:
            n = ntokens(unit.text)
            if current and current_tokens + n > size:
                body = "\n".join(current)
                emit(sec.label, body, page)
                carry = _tail(body, overlap)
                if carry and ntokens(carry) + n > size:
                    carry = ""
                current = [carry] if carry else []
                current_tokens = ntokens(carry)
                page = unit.page
            if not current:
                page = unit.page
            current.append(unit.text)
            current_tokens += n
        if current:
            emit(sec.label, "\n".join(current), page)
    return chunks


def chunk_fixed(doc: Document, size: int, overlap: int) -> list[Chunk]:
    """Fixed-size word windows that ignore structure: failure stage B on purpose."""
    words: list[tuple[str, int]] = [(w, p.page) for p in doc.pages for w in p.text.split()]
    step = max(1, size - max(0, overlap))
    out: list[Chunk] = []
    for i in range(0, len(words), step):
        window = words[i:i + size]
        if not window:
            break
        out.append(Chunk(id=f"{_slug(doc.source)}-f{len(out):04d}",
                         text=" ".join(w for w, _ in window), source=doc.source,
                         section="(fixed window)", page=window[0][1],
                         doc_version=doc.doc_version, superseded=doc.superseded))
    return out


def amplification(chunks: list[Chunk], docs: list[Document]) -> float:
    """Stored tokens divided by original tokens (the lecture's 150 / 120 example)."""
    original = sum(ntokens(p.text) for d in docs for p in d.pages)
    stored = sum(ntokens(c.text) for c in chunks)
    return round(stored / original, 3) if original else 0.0
