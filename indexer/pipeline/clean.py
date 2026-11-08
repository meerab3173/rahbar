"""Stage 2: Clean. Remove artefacts without destroying structure.

Rules (docs/PIPELINE.md lists them with examples):
  1. Running headers and footers: a line that appears, after digits are masked, on at
     least half the pages of a document with 3+ pages is removed from every page.
     ("Undergraduate Academic Rules and Regulations 2026 17" is on every page.)
  2. Bare page numbers ("17", "Page 17", "IV") on a line of their own are removed.
  3. Table-of-contents pages (dot leaders, or a front-matter page whose lines mostly end
     in a page number) are dropped: they match every query and answer none.
  4. Words hyphenated across a line break are re-joined ("regu-\\nlations").
  5. Bullet glyphs are normalised to "- ", and runs of spaces collapsed.
Deliberately NOT stripped: headings and clause numbers (chunking and citations depend on
them), pipe-separated table rows, and any text that looks like an instruction to the
model. Rule 3 of section 3.4: retrieved text is data, so a planted "ignore your
instructions" line is kept and handled at query time, not hidden at index time.
"""

from __future__ import annotations

import re
from collections import Counter

from .models import Document, Page

_PAGE_NUMBER = re.compile(r"^\s*(page\s*)?(\d{1,4}|[ivxlc]{1,7})(\s*(of|/)\s*\d+)?\s*$", re.I)
_DOT_LEADER = re.compile(r"\.{4,}\s*\d+\s*$")
_ENDS_WITH_PAGE = re.compile(r"\S\s+\d{1,3}\s*$")
_BULLETS = re.compile(r"^[\s]*[•●▪◦■□➢►]\s*", re.M)
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")


def _mask(line: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", line.strip().lower()))


def repeating_lines(pages: list[Page], min_share: float = 0.5) -> set[str]:
    if len(pages) < 3:
        return set()
    counts: Counter[str] = Counter()
    for page in pages:
        counts.update({_mask(line) for line in page.text.splitlines() if line.strip()})
    limit = max(2, int(len(pages) * min_share))
    return {line for line, n in counts.items() if n >= limit and len(line) > 3}


def is_toc_page(page: Page) -> bool:
    lines = [ln for ln in page.text.splitlines() if ln.strip()]
    if len(lines) < 4:
        return False
    if sum(1 for ln in lines if _DOT_LEADER.search(ln)) >= 3:
        return True
    ending = sum(1 for ln in lines if _ENDS_WITH_PAGE.search(ln) and len(ln) < 80)
    share = ending / len(lines)
    if "contents" in page.text.lower():
        return share >= 0.5
    # Continuation pages of a contents list, before the body starts. A real table (pipe
    # rows from extract) is never treated as a contents page.
    return page.page <= 6 and "|" not in page.text and share >= 0.7


def clean_text(text: str, drop: set[str]) -> str:
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _BULLETS.sub("- ", text)
    kept = []
    for line in text.splitlines():
        if line.strip() and (_mask(line) in drop or _PAGE_NUMBER.match(line)):
            continue
        kept.append(re.sub(r"[ \t ]+", " ", line).rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def clean(doc: Document) -> Document:
    """Strip headers and footers that repeat on every page; keep headings and
    section boundaries, because chunking and citations depend on them."""
    drop = repeating_lines(doc.pages)
    pages = []
    for page in doc.pages:
        if is_toc_page(page):
            continue
        pages.append(Page(source=page.source, page=page.page,
                          text=clean_text(page.text, drop), ocr=page.ocr))
    return Document(source=doc.source, doc_version=doc.doc_version,
                    superseded=doc.superseded, pages=pages)
