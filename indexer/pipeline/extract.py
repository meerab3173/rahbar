"""Stage 1: Extract. PDF, HTML or Markdown to text, page by page.

PDF: pdfplumber reads the text layer. A page whose text layer is empty (or nearly so)
is rasterised with pdf2image (poppler) and OCR'd with pytesseract; the page is marked
ocr=True so the loss is visible downstream. Tables found by pdfplumber are re-emitted
as pipe-separated rows ("| A+ | 4.00 |") so their columns survive chunking; the raw text
of the same region is kept as well, since pdfplumber's table finder misses borderless
tables.

Markdown: the CI fixtures. Front matter is dropped; the whole file is one page.
HTML: visible text of the page, with <table> rows emitted as pipe rows.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

from .models import Document, Page

MIN_TEXT_CHARS = 25          # fewer characters than this on a page means "no text layer"
OCR_DPI = 300


def _strip_front_matter(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        for i, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                return "\n".join(lines[i + 1:])
    return text


def table_to_rows(table: list[list[str | None]]) -> str:
    rows = []
    for row in table:
        cells = [re.sub(r"\s+", " ", c or "").strip() for c in row]
        if any(cells):
            rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def ocr_page(path: str, page_number: int) -> str:
    from pdf2image import convert_from_path   # needs poppler-utils in the image
    import pytesseract                        # needs tesseract-ocr in the image

    images = convert_from_path(path, dpi=OCR_DPI, first_page=page_number,
                               last_page=page_number)
    return "\n".join(pytesseract.image_to_string(img, lang="eng") for img in images)


def extract_pdf(path: str, ocr: bool = True) -> list[Page]:
    import pdfplumber

    source = Path(path).name
    pages: list[Page] = []
    with pdfplumber.open(path) as pdf:
        for number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(x_tolerance=1.5, y_tolerance=3) or ""
            if len(text.strip()) < MIN_TEXT_CHARS:
                if not ocr:     # failure stage A on purpose: index the scan without OCR
                    pages.append(Page(source=source, page=number, text=text, ocr=False))
                    continue
                pages.append(Page(source=source, page=number, text=ocr_page(path, number),
                                  ocr=True))
                continue
            try:
                tables = [t for t in page.extract_tables() if t and len(t) > 1]
            except Exception:   # the table finder is best-effort
                tables = []
            if tables:
                text += "\n\n" + "\n\n".join(table_to_rows(t) for t in tables)
            pages.append(Page(source=source, page=number, text=text))
    return pages


class _HTMLText(HTMLParser):
    SKIP = {"script", "style", "nav", "footer", "header", "noscript", "svg"}
    BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "section", "tr",
             "table", "article"}

    def __init__(self) -> None:
        super().__init__()
        self.out: list[str] = []
        self.skip = 0
        self.row: list[str] | None = None
        self.cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP:
            self.skip += 1
        elif tag == "tr":
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []
        elif tag in ("h1", "h2", "h3", "h4"):
            self.out.append("\n\n## ")
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in ("td", "th") and self.row is not None and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            if any(self.row):
                self.out.append("\n| " + " | ".join(self.row) + " |")
            self.row = None
        elif tag in ("h1", "h2", "h3", "h4"):
            self.out.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip:
            return
        if self.cell is not None:
            self.cell.append(data)
        elif self.row is None:
            self.out.append(data)


def extract_html(path: str) -> list[Page]:
    parser = _HTMLText()
    parser.feed(Path(path).read_text(encoding="utf-8", errors="replace"))
    text = "".join(parser.out)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return [Page(source=Path(path).name, page=1, text=text.strip())]


def extract(path: str, doc_version: str, superseded: bool, ocr: bool = True) -> Document:
    """Return the document's text page by page.

    Pages with no text layer are OCR'd and marked ocr=True (unless ocr=False, which is
    how docs/FAILURE-ANALYSIS.md breaks stage A on purpose).
    """
    suffix = Path(path).suffix.lower()
    source = Path(path).name
    if suffix == ".pdf":
        pages = extract_pdf(path, ocr=ocr)
    elif suffix in (".html", ".htm"):
        pages = extract_html(path)
    else:   # .md and .txt
        text = _strip_front_matter(Path(path).read_text(encoding="utf-8", errors="replace"))
        pages = [Page(source=source, page=1, text=text)]
    return Document(source=source, doc_version=doc_version, superseded=superseded,
                    pages=pages)
