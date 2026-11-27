"""Assignment 03, Part A: verification (A1) and incremental update (A2).

No Chroma server, no model, no network: an in-memory fake of the Chroma client and the
offline hash-bow-768 embedding, on copies of the CI fixtures in a temporary data lake.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from pipeline.config import IndexerSettings, get_settings
from pipeline.embed import embed
from pipeline.hashembed import HASH_MODEL
from pipeline.lake import DataLake, main as lake_main
from pipeline.run import derive_doc_id, ingest
from pipeline.verify import garbled_ratio, truncation, windows

FIXTURES = Path(__file__).resolve().parents[2] / "eval" / "fixtures"
GOOD_TEXT = ("Students must keep their identity card with them on campus at all times. "
             "A lost card is replaced by the admissions office within three working days. ")


# --- an in-memory Chroma ----------------------------------------------------------------------
class Col:
    def __init__(self, name: str, metadata: dict | None) -> None:
        self.name, self.metadata = name, metadata or {}
        self.rows: dict[str, tuple[str, list[float], dict]] = {}
        self.calls = {"add": 0, "delete": 0, "update": 0}

    def add(self, ids, documents, embeddings, metadatas) -> None:  # type: ignore[no-untyped-def]
        self.calls["add"] += len(ids)
        for i, d, e, m in zip(ids, documents, embeddings, metadatas):
            assert i not in self.rows, f"duplicate id {i}"
            self.rows[i] = (d, e, dict(m))

    def upsert(self, ids, embeddings, documents, metadatas) -> None:  # type: ignore[no-untyped-def]
        for i, d, e, m in zip(ids, documents, embeddings, metadatas):
            self.rows[i] = (d, e, dict(m))

    def delete(self, ids) -> None:  # type: ignore[no-untyped-def]
        self.calls["delete"] += len(ids)
        for i in ids:
            del self.rows[i]

    def update(self, ids, metadatas) -> None:  # type: ignore[no-untyped-def]
        self.calls["update"] += len(ids)
        for i, m in zip(ids, metadatas):
            d, e, old = self.rows[i]
            self.rows[i] = (d, e, {**old, **m})

    def count(self) -> int:
        return len(self.rows)

    def get(self, ids=None, include=None, limit=None, offset=0):  # type: ignore[no-untyped-def]
        keys = list(ids) if ids is not None else sorted(self.rows)
        keys = [k for k in keys if k in self.rows]
        if limit is not None:
            keys = keys[offset:offset + limit]
        return {"ids": keys, "metadatas": [self.rows[k][2] for k in keys]}


class Client:
    def __init__(self) -> None:
        self.cols: dict[str, Col] = {}

    def create_collection(self, name: str, metadata: dict | None = None) -> Col:
        assert name not in self.cols
        self.cols[name] = Col(name, metadata)
        return self.cols[name]

    def get_or_create_collection(self, name: str, metadata: dict | None = None) -> Col:
        return self.cols.get(name) or self.create_collection(name, metadata)

    def get_collection(self, name: str) -> Col:
        if name not in self.cols:
            raise ValueError(f"Collection {name} does not exist")
        return self.cols[name]

    def delete_collection(self, name: str) -> None:
        del self.cols[name]

    def list_collections(self) -> list[Col]:
        return list(self.cols.values())

    def heartbeat(self) -> int:
        return 1


# --- helpers ----------------------------------------------------------------------------------
def settings(tmp: Path, **over: Any) -> IndexerSettings:
    values = {**get_settings().__dict__, "embed_model": HASH_MODEL,
              "corpus_dir": str(tmp / "corpus"), "datalake_dir": str(tmp / "lake"),
              "chunk_size": 120, "chunk_overlap": 20, **over}
    return IndexerSettings(**values)


@pytest.fixture()
def world(tmp_path: Path) -> tuple[IndexerSettings, Client]:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for f in FIXTURES.glob("*.md"):
        if not f.name.lower().startswith("readme"):
            shutil.copy(f, corpus / f.name)
    return settings(tmp_path), Client()


class Embedder:
    """The offline embedding, counting how many texts it was asked to embed."""

    def __init__(self, s: IndexerSettings) -> None:
        self.s, self.texts = s, 0

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.texts += len(texts)
        return embed(texts, self.s)


def run(s: IndexerSettings, client: Client) -> tuple[dict[str, Any], Embedder]:
    e = Embedder(s)
    return ingest(s, client=client, embed_fn=e), e


def md(title: str | None, version: str | None, body: str, doc_id: str | None = None) -> str:
    fm = ["---"]
    if title:
        fm.append(f"title: {title}")
    if doc_id:
        fm.append(f"doc_id: {doc_id}")
    if version:
        fm.append(f"doc_version: {version}")
    fm.append("---")
    return "\n".join(fm) + "\n" + body


def drop(s: IndexerSettings, batch: str, name: str, content: str | bytes) -> None:
    folder = Path(s.datalake_dir) / "inbox" / batch
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def checks(report: dict[str, Any], file: str) -> dict[str, dict[str, str]]:
    return next(d for d in report["documents"] if d["file"] == file)["checks"]


def verify_report(s: IndexerSettings, run_id: str) -> dict[str, Any]:
    return json.loads((Path(s.datalake_dir) / "reports" / f"verify-{run_id}.json").read_text())


# --- A1: the four checks ------------------------------------------------------------------------
def test_garbled_ratio_and_windows() -> None:
    assert garbled_ratio("Plain English text, 2.00 CGPA.") == 0.0
    assert garbled_ratio("Ã©â€™Ã¢â‚¬Å“ ÃƒÂ¢Ã¢â€šÂ¬") > 0.5
    assert len(windows("x" * 10_000)) == 3 and len(windows("short")) == 1


def test_truncation_detects_cut_pdfs_and_text(tmp_path: Path) -> None:
    whole = tmp_path / "a.pdf"
    whole.write_bytes(b"%PDF-1.4\n...\n%%EOF\n")
    cut = tmp_path / "b.pdf"
    cut.write_bytes(b"%PDF-1.4\n" + b"x" * 5000)
    assert truncation(whole, "") == ""
    assert "%%EOF" in truncation(cut, "")
    text = tmp_path / "c.md"
    text.write_text("x")
    assert truncation(text, "The rule ends here.") == ""
    assert "mid-sentence" in truncation(text, "The rule says that students who")


def test_first_run_verifies_every_document_and_archives_the_old_version(world) -> None:  # type: ignore[no-untyped-def]
    s, client = world
    out, _ = run(s, client)
    rep = verify_report(s, out["run_id"])
    assert rep["counts"] == {"active": 2, "archived": 1, "quarantine": 0}
    old = checks(rep, "academic-rules-2023.md")
    assert old["recency"]["result"] == "fail"
    assert "academic-rules@2026" in old["recency"]["reason"]
    for doc in rep["documents"]:
        assert set(doc["checks"]) == {"quality", "completeness", "recency", "uniqueness"}
    lake = Path(s.datalake_dir)
    assert (lake / "archive" / "academic-rules" / "2023" / "academic-rules-2023.md").exists()
    assert (lake / "lake" / "academic-rules" / "2026" / "academic-rules-2026.md").exists()
    col = client.cols[s.collection_name]
    archived = [m for _, _, m in col.rows.values() if m["doc_key"] == "academic-rules@2023"]
    assert archived and all(m["superseded"] for m in archived)
    pointer = client.cols["rahbar-active"].rows["active"][2]
    assert pointer["collection"] == s.collection_name and pointer["generation"] == out["run_id"]


def test_batch_with_planted_problems_is_quarantined(world) -> None:  # type: ignore[no-untyped-def]
    s, client = world
    good = md("ID card rules", "2026", "# ID card rules\n\n" + GOOD_TEXT * 4, "id-card-rules")
    drop(s, "b", "01-id-card.md", good)
    drop(s, "b", "02-id-card-copy.md", good)                                    # exact dup
    drop(s, "b", "03-id-card-reworded.md",                                      # near dup
         md("ID card rules (reformatted)", "2026",
            "# ID card rules\n\n" + GOOD_TEXT * 4 + " Thank you.", "id-card-rules-copy"))
    drop(s, "b", "04-no-metadata.md", "---\n---\n" + GOOD_TEXT * 4)             # no title/date
    drop(s, "b", "05-cut.pdf", b"%PDF-1.4\n1 0 obj << /Type /Catalog >>\n" + b"0" * 900)
    drop(s, "b", "06-garbled.md",
         md("Notice", "2026", "Ã©â€™Ã¢â‚¬Å“ ÃƒÂ¢ " * 60 + "end."))
    drop(s, "b", "07-tiny.md", md("Tiny", "2026", "Too short."))
    out, _ = run(s, client)
    rep = verify_report(s, out["run_id"])
    assert checks(rep, "01-id-card.md")["uniqueness"]["result"] == "pass"
    assert "exact duplicate of 01-id-card.md" in checks(rep, "02-id-card-copy.md")["uniqueness"]["reason"]
    assert "near-duplicate of id-card-rules@2026" in \
        checks(rep, "03-id-card-reworded.md")["uniqueness"]["reason"]
    assert "title, date or version" in checks(rep, "04-no-metadata.md")["completeness"]["reason"]
    cut = checks(rep, "05-cut.pdf")
    assert cut["quality"]["result"] == "fail" and "%%EOF" in cut["completeness"]["reason"]
    assert "garbled" in checks(rep, "06-garbled.md")["quality"]["reason"]
    assert "empty" in checks(rep, "07-tiny.md")["quality"]["reason"]
    assert rep["counts"]["quarantine"] == 6
    lake = Path(s.datalake_dir)
    quarantined = {p.name for p in (lake / "quarantine" / out["run_id"]).iterdir()}
    assert "02-id-card-copy.md" in quarantined and "02-id-card-copy.md.reason.json" in quarantined
    assert not any((lake / "inbox" / "b").glob("*.md"))                         # moved out
    sources = {m["source"] for _, _, m in client.cols[s.collection_name].rows.values()}
    assert "01-id-card.md" in sources and "02-id-card-copy.md" not in sources


# --- A2: incremental update -----------------------------------------------------------------------
def test_unchanged_corpus_embeds_zero_chunks(world) -> None:  # type: ignore[no-untyped-def]
    s, client = world
    first, _ = run(s, client)
    assert first["chunks"]["embedded"] > 5
    second, embedder = run(s, client)
    assert second["chunks"]["embedded"] == 0 and second["chunks"]["deleted"] == 0
    assert embedder.texts == 0                    # not even a fingerprint was re-embedded
    assert second["chunks"]["total"] == first["chunks"]["total"]
    assert second["generation"] == first["generation"] and not second["pointer_flipped"]


def test_changed_document_replaces_only_its_own_chunks(world) -> None:  # type: ignore[no-untyped-def]
    s, client = world
    first, _ = run(s, client)
    col = client.cols[s.collection_name]
    fees_before = {i: v for i, v in col.rows.items() if v[2]["doc_id"] == "fees"}
    rules_before = {i: v for i, v in col.rows.items() if v[2]["doc_id"] == "academic-rules"}
    fees = Path(s.corpus_dir) / "fees-2026.md"
    fees.write_text(fees.read_text() + "\n## 9. Late fee\n\nA late fee of Rs 2,000 applies.\n")
    second, _ = run(s, client)
    assert second["per_document"]["fees@2026"].startswith("replaced")
    assert 0 < second["chunks"]["embedded"] <= len(fees_before) + 1
    assert second["chunks"]["deleted"] == len(fees_before)
    rules_after = {i: v for i, v in col.rows.items() if v[2]["doc_id"] == "academic-rules"}
    assert rules_after == rules_before                        # untouched, same vectors
    assert second["generation"] != first["generation"] and second["pointer_flipped"]


def test_deleted_document_leaves_no_orphan_chunks(world) -> None:  # type: ignore[no-untyped-def]
    s, client = world
    first, _ = run(s, client)
    assert first["counts_by_doc_id"]["fees"] > 0
    (Path(s.corpus_dir) / "fees-2026.md").unlink()
    second, _ = run(s, client)
    assert "fees" not in second["counts_by_doc_id"]
    assert not any(m["doc_id"] == "fees"
                   for _, _, m in client.cols[s.collection_name].rows.values())
    assert second["orphans"] == []
    assert not (Path(s.datalake_dir) / "lake" / "fees").exists()


def test_restore_rolls_back_to_the_archived_version_without_embedding(world) -> None:  # type: ignore[no-untyped-def]
    s, client = world
    run(s, client)
    DataLake(s.datalake_dir).request("restore", "academic-rules@2023")
    out, embedder = run(s, client)
    assert embedder.texts == 0 and out["chunks"]["embedded"] == 0
    assert out["chunks"]["relabelled"] > 0
    rows = client.cols[s.collection_name].rows.values()
    by_key = {m["doc_key"]: m["superseded"] for _, _, m in rows}
    assert by_key["academic-rules@2023"] is False and by_key["academic-rules@2026"] is True
    rep = verify_report(s, out["run_id"])
    assert "pinned" in checks(rep, "academic-rules-2026.md")["recency"]["reason"]
    lake = Path(s.datalake_dir)
    assert (lake / "lake" / "academic-rules" / "2023" / "academic-rules-2023.md").exists()
    assert (lake / "archive" / "academic-rules" / "2026" / "academic-rules-2026.md").exists()
    # a later run keeps the rollback until it is released
    again, _ = run(s, client)
    assert again["chunks"]["relabelled"] == 0
    DataLake(s.datalake_dir).request("release", "academic-rules")
    forward, _ = run(s, client)
    rows = client.cols[s.collection_name].rows.values()
    assert {m["doc_key"]: m["superseded"] for _, _, m in rows}["academic-rules@2026"] is False
    assert forward["chunks"]["embedded"] == 0


def test_delete_request_removes_a_batch_document(world) -> None:  # type: ignore[no-untyped-def]
    s, client = world
    drop(s, "b", "id-card.md",
         md("ID card rules", "2026", "# ID card rules\n\n" + GOOD_TEXT * 4, "id-card-rules"))
    first, _ = run(s, client)
    assert first["counts_by_doc_id"]["id-card-rules"] > 0
    second, _ = run(s, client)                       # carried over from the lake, unchanged
    assert second["chunks"]["embedded"] == 0
    DataLake(s.datalake_dir).request("delete", "id-card-rules@2026")
    third, _ = run(s, client)
    assert "id-card-rules" not in third["counts_by_doc_id"] and third["orphans"] == []
    assert third["chunks"]["deleted"] == first["counts_by_doc_id"]["id-card-rules"]


def test_derive_doc_id_drops_the_version() -> None:
    assert derive_doc_id("academic-rules-2023.md", "2023") == "academic-rules"
    assert derive_doc_id("Notice.PDF", "") == "notice"


def test_lake_cli_status_and_report(world, capsys) -> None:  # type: ignore[no-untyped-def]
    s, client = world
    run(s, client)
    assert lake_main(["--root", s.datalake_dir, "status"]) == 0
    assert "academic-rules@2023" in capsys.readouterr().out
    assert lake_main(["--root", s.datalake_dir, "report", "verify"]) == 0
    assert "VERIFICATION REPORT" in capsys.readouterr().out
    assert lake_main(["--root", s.datalake_dir, "request", "restore", "fees@2026"]) == 0
