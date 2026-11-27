"""The ingestion Job's entry point. It runs once and exits (ADR 0001).

    python -m pipeline.run --dummy    # Assignment 02 week 1: three fake chunks
    python -m pipeline.run            # ingest -> verify -> update the data lake ->
                                      # chunk -> embed what changed -> sync the store

Figure 3 of the RAGOps paper, as this Job runs it (Assignment 03, Part A):

  1. ingest     the fetched corpus (sources/), documents already in the lake, and new
                batches in inbox/<batch>/; queued control requests are applied first
  2. verify     quality, completeness, recency, uniqueness (pipeline.verify); a
                verification report per run
  3. update the data lake
                passed -> lake/, older versions -> archive/, failed -> quarantine/;
                state/catalog.json records every version
  4. update the retrieval sources
                incremental sync of the vector store to the catalog (pipeline.sync):
                unchanged documents embed nothing, a changed one replaces only its own
                chunks, a deleted one leaves no chunk behind

Exit code 0 means the run completed, even if some documents were quarantined (that is
the verification stage working). Anything else fails the Job, and Kubernetes retries it
up to the Job's backoffLimit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from .config import IndexerSettings, get_settings
from .lake import Catalog, DataLake, Entry, new_run_id
from .models import Chunk, Document
from .verify import InputDoc, Verdict, report, verify_all

SKIP_FILES = {"_index.json", "_meta.yaml", "_meta.yml"}
DOC_SUFFIXES = {".pdf", ".md", ".html", ".htm", ".txt"}


def log(msg: str, **fields: object) -> None:
    print(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "logger": "rahbar.indexer", "msg": msg, **fields}), flush=True)


def _front_matter(path: Path) -> dict[str, str]:
    """Read `key: value` lines between leading '---' markers (Markdown documents)."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    meta: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, _, value = line.partition(":")
        if key.strip():
            meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta


def discover(corpus_dir: str) -> list[tuple[Path, str, bool]]:
    """List (path, doc_version, superseded) for every document in a corpus folder.

    Kept for eval/tune_chunking.py and eval/list_sections.py. corpus/fetch.py writes
    _index.json next to the files it downloads; without it, Markdown front matter.
    """
    return [(i.path, i.version, bool(i.meta.get("superseded")))
            for i in source_inputs(corpus_dir)]


# --- 1. ingest: what this run looks at -----------------------------------------------------------
def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "yes", "1"}


def derive_doc_id(name: str, version: str) -> str:
    """A doc_id when the metadata gives none: the file name without its version."""
    stem = Path(name).stem.lower()
    if version:
        stem = stem.replace(version.lower(), " ")
    return re.sub(r"[^a-z0-9]+", "-", stem).strip("-") or "document"


def _first_heading(path: Path) -> str:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _embedded_metadata(path: Path) -> dict[str, Any]:
    """Title and date a file carries itself: PDF Info, HTML <title>, Markdown front matter
    (or its first '# ' heading). Used for documents nobody described in a sidecar."""
    suffix = path.suffix.lower()
    meta: dict[str, Any] = {}
    try:
        if suffix == ".md":
            fm = _front_matter(path)
            meta.update({k: v for k, v in fm.items() if k in
                         {"title", "doc_id", "doc_version", "date", "superseded", "source"}})
            if "title" not in meta and fm:
                heading = _first_heading(path)
                if heading:
                    meta["title"] = heading
        elif suffix == ".pdf":
            import pdfplumber

            with pdfplumber.open(str(path)) as pdf:
                info = pdf.metadata or {}
            if info.get("Title"):
                meta["title"] = str(info["Title"]).strip()
            raw_date = str(info.get("ModDate") or info.get("CreationDate") or "")
            found = re.search(r"(\d{4})(\d{2})(\d{2})", raw_date)
            if found:
                meta["date"] = "-".join(found.groups())
        elif suffix in (".html", ".htm"):
            text = path.read_text(encoding="utf-8", errors="replace")
            found = re.search(r"<title[^>]*>(.*?)</title>", text, re.I | re.S)
            if found and found.group(1).strip():
                meta["title"] = " ".join(found.group(1).split())
    except Exception:       # a corrupt file: completeness and quality will say so
        pass
    return meta


def _input(path: Path, origin: str, meta: dict[str, Any], move: bool) -> InputDoc:
    meta = dict(meta)
    version = str(meta.get("doc_version") or meta.get("date") or "")
    meta["superseded"] = _bool(meta.get("superseded"))
    if not meta.get("doc_id"):
        meta["doc_id"] = derive_doc_id(path.name, version)
    return InputDoc(path=path, origin=origin, file=path.name, sha256=sha256_of(path),
                    meta=meta, move=move)


def source_inputs(corpus_dir: str) -> list[InputDoc]:
    """The fetched corpus: corpus/fetch.py's _index.json (from corpus/sources.yaml), or the
    Markdown front matter of the CI fixtures. Copied into the lake, never moved."""
    root = Path(corpus_dir)
    if not root.exists():
        return []
    index = root / "_index.json"
    found: list[InputDoc] = []
    if index.exists():
        for e in json.loads(index.read_text(encoding="utf-8")):
            path = root / e["file"]
            if path.exists():
                meta = {"title": e.get("title") or e.get("name"), "doc_id": e.get("doc_id"),
                        "doc_version": e.get("doc_version"), "superseded": e.get("superseded"),
                        "url": e.get("url")}
                found.append(_input(path, "sources", meta, move=False))
            else:
                log("document missing, skipped", source=e["file"])
        return found
    for path in sorted(root.iterdir()):
        if path.suffix.lower() not in DOC_SUFFIXES or path.name.lower().startswith("readme"):
            continue
        found.append(_input(path, "sources", _embedded_metadata(path), move=False))
    return found


def inbox_inputs(lake: DataLake) -> list[InputDoc]:
    """New batches: inbox/<batch>/<file>. Metadata from inbox/<batch>/_meta.yaml
    (file -> fields), else what the file carries itself. Moved out of the inbox."""
    found: list[InputDoc] = []
    if not lake.inbox.exists():
        return found
    for batch in sorted(p for p in lake.inbox.iterdir() if p.is_dir()):
        sidecar: dict[str, Any] = {}
        for name in ("_meta.yaml", "_meta.yml"):
            if (batch / name).exists():
                sidecar = yaml.safe_load((batch / name).read_text(encoding="utf-8")) or {}
        for path in sorted(batch.iterdir()):
            if (not path.is_file() or path.name in SKIP_FILES or path.name.startswith(".")
                    or path.name.lower().startswith("readme")):
                continue
            meta = _embedded_metadata(path)
            meta.update(sidecar.get(path.name) or {})
            found.append(_input(path, f"batch:{batch.name}", meta, move=True))
    return found


def lake_inputs(lake: DataLake, catalog: Catalog) -> list[InputDoc]:
    """Documents an earlier batch put in the lake: they stay part of the corpus."""
    found = []
    for entry in sorted(catalog.entries.values(), key=lambda e: e.doc_key):
        if entry.origin == "sources":
            continue
        path = lake.abs(entry.path)
        if not path.exists():
            log("lake file missing; the document is dropped", doc_key=entry.doc_key,
                path=entry.path)
            continue
        meta = {"title": entry.title, "doc_id": entry.doc_id, "doc_version": entry.version,
                "superseded": False}
        doc = InputDoc(path=path, origin=entry.origin, file=entry.file, sha256=entry.sha256,
                       meta=meta, move=False, entry=entry)
        found.append(doc)
    return found


def extract_document(inp: InputDoc, settings: IndexerSettings) -> tuple[Document | None, str]:
    from .clean import clean
    from .extract import extract

    try:
        raw = extract(str(inp.path), inp.version or "unversioned", False, ocr=settings.ocr)
        doc = clean(raw)
    except Exception as exc:  # corrupt or truncated file: the quality check reports it
        return None, f"{type(exc).__name__}: {str(exc)[:160]}"
    doc.doc_id = inp.doc_id
    doc.title = str(inp.meta.get("title") or "")
    doc.sha256 = inp.sha256
    return doc, ""


# --- control requests ----------------------------------------------------------------------------
def apply_requests(lake: DataLake, catalog: Catalog) -> list[str]:
    outcomes = []
    for path, req in lake.pending_requests():
        action, target = req.get("action", ""), req.get("target", "")
        if action == "restore":
            entry = catalog.entries.get(target)
            if entry is None:
                outcome = f"ignored: no document version {target}"
            else:
                for other in catalog.by_doc_id(entry.doc_id):
                    other.pinned = other is entry
                outcome = f"restore {target}: pinned as the current version of {entry.doc_id}"
        elif action == "release":
            pinned = [e for e in catalog.by_doc_id(target) if e.pinned]
            for e in pinned:
                e.pinned = False
            outcome = (f"release {target}: recency decides again" if pinned
                       else f"ignored: nothing pinned for {target}")
        elif action == "delete":
            entry = catalog.entries.pop(target, None)
            if entry is None:
                outcome = f"ignored: no document version {target}"
            else:
                catalog.deleted[target] = entry.sha256
                lake.remove(entry)
                outcome = f"delete {target}: removed from the lake; its chunks go this run"
        else:
            outcome = f"ignored: unknown action {action!r}"
        lake.mark_applied(path, outcome)
        outcomes.append(outcome)
        log("control request", action=action, target=target, outcome=outcome)
    return outcomes


# --- the run --------------------------------------------------------------------------------------
def ingest(settings: IndexerSettings, client: Any = None,
           embed_fn: Any = None) -> dict[str, Any]:
    from .embed import embed
    from .store import wait_for_chroma
    from .sync import sync

    started = time.perf_counter()
    run_id = new_run_id()
    lake = DataLake(settings.datalake_dir)
    lake.ensure()
    catalog = lake.load_catalog()
    previous = dict(catalog.entries)
    embed_fn = embed_fn or (lambda texts: embed(texts, settings))

    # 1. ingest
    requests = apply_requests(lake, catalog)
    known = dict(catalog.entries)
    inputs = source_inputs(settings.corpus_dir) + lake_inputs(lake, catalog) + \
        inbox_inputs(lake)
    skipped = []
    for inp in list(inputs):
        if catalog.deleted.get(inp.doc_key) == inp.sha256:
            inputs.remove(inp)
            skipped.append(inp.doc_key)
            if inp.move:
                inp.path.unlink()
    log("ingest", run_id=run_id, documents=len(inputs), requests=len(requests),
        skipped_deleted=skipped)

    # 2. verify
    verdicts = []
    for inp in inputs:
        document, error = extract_document(inp, settings)
        verdicts.append(Verdict(input=inp, document=document, error=error))
    fingerprints = lake.load_json("fingerprints.json")
    verify_all(verdicts, settings, embed_fn, fingerprints, known)
    lake.save_json("fingerprints.json", fingerprints)

    # 3. update the data lake
    final: dict[str, Entry] = {}
    placed: dict[str, str] = {}
    documents: dict[str, Document] = {}
    for v in verdicts:
        inp = v.input
        tag = inp.doc_key + "|" + inp.file
        if v.decision == "quarantine":
            placed[tag] = lake.to_quarantine(inp.path, run_id, {
                "doc_key": inp.doc_key, "origin": inp.origin, "checks": v.checks},
                move=inp.move)
            continue
        prev = previous.get(inp.doc_key)
        recency = v.checks["recency"]
        entry = Entry(doc_key=inp.doc_key, doc_id=inp.doc_id, version=inp.version,
                      title=str(inp.meta.get("title") or ""), file=inp.file,
                      origin=inp.origin, path=prev.path if prev else "", sha256=inp.sha256,
                      pages=len(v.document.pages) if v.document else 0, status=v.decision,
                      pinned=bool(known.get(inp.doc_key) and known[inp.doc_key].pinned),
                      reason=recency["reason"] if v.decision == "archived" else "",
                      first_run=prev.first_run if prev else run_id, last_run=run_id)
        if inp.entry is not None:                          # already in the lake
            entry.path = lake.relocate(entry)
        else:
            entry.path = lake.place(inp.path, entry, move=inp.move)
            if prev and prev.path and prev.path != entry.path:
                lake.remove(prev)          # moved between lake/ and archive/, or replaced
        placed[tag] = entry.path
        final[entry.doc_key] = entry
        if v.document is not None:
            documents[entry.doc_key] = v.document
    # a known version whose new bytes failed verification keeps its last good copy
    for v in verdicts:
        prev = previous.get(v.input.doc_key)
        if v.decision == "quarantine" and prev and v.input.doc_key not in final:
            final[prev.doc_key] = prev
    for key, prev in previous.items():
        if key not in final:
            lake.remove(prev)                              # deleted: the file goes too
    # an emptied batch folder (only its README or _meta.yaml left) is filed away
    for batch in {v.input.path.parent for v in verdicts if v.input.move}:
        if batch.exists() and not any(p.suffix.lower() in DOC_SUFFIXES and
                                      not p.name.lower().startswith("readme")
                                      for p in batch.iterdir()):
            done = lake.state / "processed-batches" / f"{run_id}-{batch.name}"
            done.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(batch), str(done))
    catalog.entries = final
    lake.save_catalog(catalog)
    verify_data, verify_text = report(run_id, verdicts, settings, placed)
    lake.write_report("verify", run_id, verify_data, verify_text)
    print(verify_text, flush=True)

    # 4. update the retrieval sources
    client = client or wait_for_chroma(settings)
    desired = [(entry, documents.get(key)) for key, entry in sorted(final.items())]
    result = sync(client, desired, settings, embed_fn, run_id)
    seconds = round(time.perf_counter() - started, 1)
    ingest_data = {
        "run_id": run_id, "collection": result.collection, "generation": result.generation,
        "pointer_flipped": result.pointer_flipped, "requests": requests,
        "skipped_deleted": skipped, "documents": verify_data["counts"],
        "chunks": {"embedded": result.embedded, "deleted": result.deleted,
                   "kept": result.kept, "relabelled": result.relabelled,
                   "total": result.total},
        "per_document": result.per_doc, "counts_by_doc_id": result.counts_by_doc_id,
        "counts_by_doc_key": result.counts_by_doc_key, "orphans": result.orphans,
        "seconds": seconds, "embed_model": settings.embed_model,
        "chunk_size": settings.chunk_size, "chunk_overlap": settings.chunk_overlap}
    ingest_text = ingest_report_text(ingest_data)
    lake.write_report("ingest", run_id, ingest_data, ingest_text)
    print(ingest_text, flush=True)
    return ingest_data


def ingest_report_text(d: dict[str, Any]) -> str:
    c = d["chunks"]
    lines = [f"INGESTION REPORT  run {d['run_id']}",
             f"collection {d['collection']}  generation {d['generation'] or '-'}"
             f"{'  (pointer flipped)' if d['pointer_flipped'] else ''}",
             f"model {d['embed_model']}  chunks {d['chunk_size']}/{d['chunk_overlap']}", ""]
    if d["requests"]:
        lines += ["requests applied:"] + [f"  {r}" for r in d["requests"]] + [""]
    lines.append("per document:")
    lines += [f"  {k:<58} {v}" for k, v in sorted(d["per_document"].items())]
    lines += ["", "chunks by doc_id (after this run):"]
    lines += [f"  {k:<58} {n:>5}" for k, n in d["counts_by_doc_id"].items()]
    lines += ["", f"CHUNKS EMBEDDED: {c['embedded']}   deleted: {c['deleted']}   "
                  f"kept: {c['kept']}   relabelled: {c['relabelled']}   "
                  f"total in collection: {c['total']}",
              f"orphan chunks (doc_keys not in the catalog): {len(d['orphans'])}",
              f"seconds: {d['seconds']}"]
    return "\n".join(lines) + "\n"


# --- Assignment 02 helpers, still used by eval/ tools -------------------------------------------
def build_chunks(settings: IndexerSettings) -> tuple[list[Chunk], float]:
    """Extract, clean and chunk a corpus folder without the lake (offline tuning tools)."""
    from .chunk import amplification, chunk

    all_chunks: list[Chunk] = []
    cleaned = []
    for inp in source_inputs(settings.corpus_dir):
        doc, error = extract_document(inp, settings)
        if doc is None:
            log("extraction failed, skipped", source=inp.file, error=error)
            continue
        doc.superseded = bool(inp.meta.get("superseded"))
        cleaned.append(doc)
        all_chunks.extend(chunk(doc, settings.chunk_size, settings.chunk_overlap,
                                strategy=settings.chunk_strategy))
    if not cleaned:
        raise SystemExit(f"no documents found in {settings.corpus_dir}")
    return all_chunks, amplification(all_chunks, cleaned)


def run_pipeline(settings: IndexerSettings, client: Any = None) -> int:
    """The whole ingestion run; returns the number of chunks in the collection."""
    return int(ingest(settings, client=client)["chunks"]["total"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest, verify and index, then exit.")
    parser.add_argument("--dummy", action="store_true", help="write three fake chunks")
    args = parser.parse_args()
    settings = get_settings()
    log("indexer starting", dummy=args.dummy, embed_model=settings.embed_model,
        collection=settings.collection_name, datalake=settings.datalake_dir,
        chroma=f"{settings.chroma_host}:{settings.chroma_port}",
        pod=os.environ.get("HOSTNAME", "-"))
    started = time.perf_counter()
    try:
        if args.dummy:
            from .dummy import write_dummy_index
            from .store import wait_for_chroma

            count = write_dummy_index(settings, wait_for_chroma(settings))
        else:
            count = run_pipeline(settings)
    except Exception as exc:  # a non-zero exit fails the Job, and backoffLimit retries it
        log("indexer failed", error=f"{type(exc).__name__}: {exc}")
        return 1
    log("indexer finished", chunks=count, seconds=round(time.perf_counter() - started, 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
