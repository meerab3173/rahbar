"""The shapes passed between stages. Metadata fields are required (section 3.1)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Page:
    """Text of one page, as extraction produced it."""

    source: str              # file name, e.g. Academic_Rules_Handbook_2026.pdf
    page: int                # 1-based
    text: str
    ocr: bool = False        # True when this page had no text layer and was OCR'd


@dataclass
class Document:
    source: str
    doc_version: str         # e.g. "2026-06"; from corpus/sources.yaml
    superseded: bool         # True for an older version of a policy
    pages: list[Page] = field(default_factory=list)
    # Data-lake identity (Assignment 03, Part A). doc_id names the document across its
    # versions ("academic-rules-handbook"); doc_key is one version of it
    # ("academic-rules-handbook@2026-06"); sha256 is of the original file's bytes.
    doc_id: str = ""
    title: str = ""
    sha256: str = ""

    @property
    def doc_key(self) -> str:
        return f"{self.doc_id or self.source}@{self.doc_version}"


@dataclass
class Chunk:
    id: str                  # stable: same document + same position gives the same id
    text: str
    source: str
    section: str             # the heading this chunk sits under, e.g. "5.4 Academic Warning"
    page: int
    doc_version: str
    superseded: bool
    doc_id: str = ""         # the document across versions (Part A): counted per doc_id
    doc_sha256: str = ""     # the file this chunk came from: unchanged hash = no re-embed

    @property
    def doc_key(self) -> str:
        return f"{self.doc_id or self.source}@{self.doc_version}"

    def metadata(self) -> dict[str, str | int | bool]:
        """Chroma metadata values must be str, int, float or bool."""
        return {
            "source": self.source,
            "section": self.section,
            "page": self.page,
            "doc_version": self.doc_version,
            "superseded": self.superseded,
            "doc_id": self.doc_id or self.source,
            "doc_key": self.doc_key,
            "doc_sha256": self.doc_sha256,
        }
