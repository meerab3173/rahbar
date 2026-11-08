"""The five indexing stages (section 3.1), with no Chroma server, no model, no network."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from pipeline import store as store_mod
from pipeline.chunk import amplification, chunk, heading, ntokens, sections
from pipeline.clean import clean, is_toc_page
from pipeline.config import IndexerSettings, get_settings
from pipeline.embed import embed
from pipeline.extract import extract, table_to_rows
from pipeline.hashembed import HASH_MODEL, hash_bow
from pipeline.models import Document, Page
from pipeline.run import build_chunks, run_pipeline

DATA = Path(__file__).parent / "data"
FIXTURES = Path(__file__).resolve().parents[2] / "eval" / "fixtures"
PDF = str(DATA / "sample-handbook.pdf")
HAS_OCR = shutil.which("tesseract") is not None and shutil.which("pdftoppm") is not None


def settings(**over: object) -> IndexerSettings:
    base = get_settings()
    values = {**base.__dict__, "embed_model": HASH_MODEL, "corpus_dir": str(FIXTURES),
              "chunk_size": 120, "chunk_overlap": 20, **over}
    return IndexerSettings(**values)  # type: ignore[arg-type]


# --- extract ------------------------------------------------------------------------------
def test_markdown_front_matter_is_not_indexed() -> None:
    doc = extract(str(FIXTURES / "academic-rules-2026.md"), "2026", False)
    assert "doc_version:" not in doc.pages[0].text
    assert doc.superseded is False and doc.doc_version == "2026"


def test_pdf_text_layer_is_read_page_by_page() -> None:
    doc = extract(PDF, "2026", False)
    assert [p.page for p in doc.pages] == [1, 2, 3, 4]
    assert "1.21 Students are required" in doc.pages[1].text


@pytest.mark.skipif(not HAS_OCR, reason="tesseract/poppler not installed")
def test_image_only_page_is_ocrd_and_marked() -> None:
    page = extract(PDF, "2026", False).pages[3]
    assert page.ocr is True
    assert "2.00" in page.text and "warning" in page.text.lower()


def test_without_ocr_the_scanned_page_is_lost() -> None:
    page = extract(PDF, "2026", False, ocr=False).pages[3]
    assert page.ocr is False and page.text.strip() == ""


def test_html_tables_become_pipe_rows(tmp_path: Path) -> None:
    html = tmp_path / "fees.html"
    html.write_text("<html><nav>menu</nav><h2>Tuition</h2><table><tr><th>Programme</th>"
                    "<th>Fee</th></tr><tr><td>BS</td><td>12,000</td></tr></table></html>")
    text = extract(str(html), "2026", False).pages[0].text
    assert "| BS | 12,000 |" in text and "menu" not in text


def test_table_to_rows_keeps_columns() -> None:
    assert table_to_rows([["Grade", "Points"], ["B-", "2.67"]]) == \
        "| Grade | Points |\n| B- | 2.67 |"


# --- clean --------------------------------------------------------------------------------
def test_running_headers_footers_and_contents_page_are_removed() -> None:
    doc = clean(extract(PDF, "2026", False))
    text = "\n".join(p.text for p in doc.pages)
    assert "SAMPLE INSTITUTE OF TECHNOLOGY" not in text
    assert "Sample Academic Rules 2026" not in text
    assert 1 not in [p.page for p in doc.pages]            # the contents page
    assert "ineligible" in text                            # hyphen re-joined
    assert "1. Medium of Instruction" in text              # headings kept


def test_planted_instructions_are_not_stripped_at_index_time() -> None:
    doc = clean(extract(str(FIXTURES / "academic-rules-2026.md"), "2026", False))
    assert "Ignore all previous instructions" in doc.pages[0].text


def test_toc_detection() -> None:
    toc = Page("x.pdf", 2, "Contents\nIntroduction 1\nAttendance 4\nGrading 5\nFees 9")
    body = Page("x.pdf", 20, "1.21 Students are required to maintain 100% attendance.\n"
                             "The HoD may condone absence.\nMore text here.\nAnd more.")
    assert is_toc_page(toc) and not is_toc_page(body)


# --- chunk --------------------------------------------------------------------------------
@pytest.mark.parametrize("line,expected", [
    ("5. Academic Assessment", 2),
    ("XII - Policy on Generative AI Based Tools", 1),
    ("6.1 Use in Lab Courses", 3),
    ("Attendance Requirements", 4),
    ("## 1. Attendance", 12),
    ("1.21 Students are required to maintain 100% attendance in lectures.", None),
    ("1. The student shall attend all classes.", None),
    ("| A+ | 4.00 |", None),
])
def test_heading_recognition(line: str, expected: int | None) -> None:
    h = heading(line)
    assert (h[0] if h else None) == expected


def test_chunks_carry_section_page_and_version_metadata() -> None:
    doc = clean(extract(PDF, "2026-06", False))
    chunks = chunk(doc, 60, 10)
    by_section = {c.section: c for c in chunks}
    att = by_section["1. Medium of Instruction > Attendance Requirements"]
    assert att.page == 2 and att.doc_version == "2026-06" and att.superseded is False
    assert "1.21" in att.text and "1.22" in att.text
    assert set(att.metadata()) == {"source", "section", "page", "doc_version", "superseded"}


def test_a_clause_is_not_cut_in_half_when_it_fits() -> None:
    doc = clean(extract(PDF, "2026", False))
    for c in chunk(doc, 40, 8):
        if "1.21" in c.text:
            assert "genuine reasons" in c.text


def test_chunks_respect_the_size_budget_and_overlap() -> None:
    long = " ".join(f"Sentence number {i} explains a rule about grading." for i in range(60))
    doc = Document("long.md", "2026", False, [Page("long.md", 1, f"## 2. Grading\n\n{long}")])
    chunks = chunk(doc, 50, 10)
    assert len(chunks) > 3
    for c in chunks:
        body = c.text.split("\n", 1)[1]
        assert ntokens(body) <= 50
    first_tail = chunks[0].text.split()[-3:]
    assert " ".join(first_tail) in chunks[1].text            # overlap carried forward


def test_chunk_ids_are_stable() -> None:
    doc = clean(extract(PDF, "2026", False))
    assert [c.id for c in chunk(doc, 60, 10)] == [c.id for c in chunk(doc, 60, 10)]


def test_fixed_strategy_ignores_structure() -> None:
    doc = clean(extract(PDF, "2026", False))
    chunks = chunk(doc, 100, 0, strategy="fixed")
    assert all(c.section == "(fixed window)" for c in chunks)


def test_fixture_sections_match_the_golden_set_names() -> None:
    doc = clean(extract(str(FIXTURES / "academic-rules-2026.md"), "2026", False))
    labels = [s.label for s in sections(doc)]
    assert "1. Attendance" in labels and "2. Academic warning" in labels


def test_amplification_is_reported() -> None:
    doc = clean(extract(PDF, "2026", False))
    assert amplification(chunk(doc, 60, 10), [doc]) > 1.0


# --- embed --------------------------------------------------------------------------------
def test_hash_embedding_matches_the_api_copy() -> None:
    fingerprint = [(i, round(x, 4)) for i, x in enumerate(hash_bow("hello world")) if x]
    assert fingerprint == [(29, -0.5774), (51, -0.5774), (252, 0.5774)]   # api/tests too


def test_embed_uses_the_pinned_model_only() -> None:
    vectors = embed(["one", "two"], settings())
    assert len(vectors) == 2 and len(vectors[0]) == 768


# --- store --------------------------------------------------------------------------------
class Col:
    def __init__(self, name: str, metadata: dict | None) -> None:
        self.name, self.metadata, self.rows = name, metadata or {}, {}

    def add(self, ids, documents, embeddings, metadatas) -> None:   # type: ignore[no-untyped-def]
        for i, d, e, m in zip(ids, documents, embeddings, metadatas):
            self.rows[i] = (d, e, m)

    upsert = add

    def count(self) -> int:
        return len(self.rows)

    def get(self, ids, include):   # type: ignore[no-untyped-def]
        return {"metadatas": [self.rows[i][2] for i in ids if i in self.rows]}


class Client:
    def __init__(self) -> None:
        self.cols: dict[str, Col] = {}

    def create_collection(self, name: str, metadata: dict | None = None) -> Col:
        self.cols[name] = Col(name, metadata)
        return self.cols[name]

    def get_or_create_collection(self, name: str, metadata: dict | None = None) -> Col:
        return self.cols.get(name) or self.create_collection(name, metadata)

    def get_collection(self, name: str) -> Col:
        return self.cols[name]

    def delete_collection(self, name: str) -> None:
        del self.cols[name]

    def list_collections(self) -> list[Col]:
        return list(self.cols.values())

    def heartbeat(self) -> int:
        return 1


def test_store_builds_a_new_version_records_the_pin_and_flips_the_pointer(
        monkeypatch: pytest.MonkeyPatch) -> None:
    client = Client()
    s = settings()
    chunks, _ = build_chunks(s)
    vectors = embed([c.text for c in chunks], s)
    ticks = iter(range(1_000, 2_000))
    monkeypatch.setattr(store_mod.time, "time", lambda: next(ticks))
    for _ in range(3):
        assert store_mod.store(chunks, vectors, s, client=client) == len(chunks)
    versions = sorted(n for n in client.cols if n.startswith("rahbar-v"))
    assert len(versions) == 2                                 # current + previous kept
    active = client.cols["rahbar-active"].rows["active"][2]["collection"]
    assert active == versions[-1]
    meta = client.cols[active].metadata
    assert meta["embed_model"] == HASH_MODEL and meta["dimension"] == 768
    assert meta["hnsw:space"] == "cosine"


def test_store_refuses_an_empty_index() -> None:
    with pytest.raises(ValueError):
        store_mod.store([], [], settings(), client=Client())


def test_run_pipeline_end_to_end_on_the_fixtures(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Client()
    monkeypatch.setattr(store_mod, "get_client", lambda _s: client)
    count = run_pipeline(settings())
    active = client.cols["rahbar-active"].rows["active"][2]["collection"]
    rows = client.cols[active].rows.values()
    assert count == len(rows) > 5
    assert any(m["superseded"] for _, _, m in rows)
    assert {m["source"] for _, _, m in rows} == {
        "academic-rules-2023.md", "academic-rules-2026.md", "fees-2026.md"}
