"""The data lake (Assignment 03, Part A; RAGOps paper 4.2.1 to 4.2.4).

Everything the ingestion Job knows about documents lives on one PersistentVolumeClaim,
mounted at DATALAKE_DIR (/datalake), so it survives pod restarts:

    sources/      the fetched corpus (corpus/fetch.py writes here; a download cache)
    inbox/<batch>/  new documents waiting for the next run (kubectl cp puts them here)
    lake/<doc_id>/<version>/<file>      verified, current documents
    archive/<doc_id>/<version>/<file>   verified, superseded versions (kept, restorable)
    quarantine/<run_id>/<file>          documents that failed verification, with
                                        <file>.reason.json saying which check and why
    state/catalog.json                  one record per document version (the source of
                                        truth the vector store is synced to)
    state/fingerprints.json             cached document fingerprints (near-duplicates)
    control/                            requests (restore, delete, release) that the next
                                        run applies; the Job stays the only writer
    reports/                            verify-<run_id>.json/.txt and ingest-<run_id>.json/.txt

Command line (inside the datalake-shell pod, or locally with DATALAKE_DIR set):

    python -m pipeline.lake status
    python -m pipeline.lake request restore academic-rules-handbook@2025-06
    python -m pipeline.lake request delete library-borrowing-rules@2026-09
    python -m pipeline.lake request release academic-rules-handbook
    python -m pipeline.lake report verify          # the latest verification report
    python -m pipeline.lake report ingest <run_id>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ACTIONS = ("restore", "delete", "release")


@dataclass
class Entry:
    """One version of one document, as the catalog records it."""

    doc_key: str            # "<doc_id>@<version>"
    doc_id: str
    version: str
    title: str
    file: str               # original file name
    origin: str             # "sources" or "batch:<name>"
    path: str               # where the file is now, relative to the lake root
    sha256: str
    pages: int
    status: str             # "active" | "archived"
    pinned: bool = False    # set by a restore: recency does not archive it
    reason: str = ""        # why it is archived, if it is
    first_run: str = ""
    last_run: str = ""

    @property
    def superseded(self) -> bool:
        return self.status == "archived"


@dataclass
class Catalog:
    entries: dict[str, Entry] = field(default_factory=dict)
    # doc_key -> sha256 of documents deleted on request: the same bytes are skipped
    # if they come back (for example from the fetch cache).
    deleted: dict[str, str] = field(default_factory=dict)

    def by_doc_id(self, doc_id: str) -> list[Entry]:
        return [e for e in self.entries.values() if e.doc_id == doc_id]


class DataLake:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.sources = self.root / "sources"
        self.inbox = self.root / "inbox"
        self.lake = self.root / "lake"
        self.archive = self.root / "archive"
        self.quarantine = self.root / "quarantine"
        self.state = self.root / "state"
        self.control = self.root / "control"
        self.reports = self.root / "reports"

    def ensure(self) -> None:
        for d in (self.inbox, self.lake, self.archive, self.quarantine, self.state,
                  self.control, self.control / "applied", self.reports):
            d.mkdir(parents=True, exist_ok=True)

    def rel(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.root.resolve()))
        except ValueError:
            return str(path)

    def abs(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.root / p

    # --- the catalog ----------------------------------------------------------------------
    def load_catalog(self) -> Catalog:
        path = self.state / "catalog.json"
        if not path.exists():
            return Catalog()
        data = json.loads(path.read_text(encoding="utf-8"))
        return Catalog(entries={k: Entry(**v) for k, v in data.get("entries", {}).items()},
                       deleted=dict(data.get("deleted", {})))

    def save_catalog(self, catalog: Catalog) -> None:
        self.state.mkdir(parents=True, exist_ok=True)
        data = {"entries": {k: asdict(v) for k, v in sorted(catalog.entries.items())},
                "deleted": catalog.deleted}
        _atomic_write(self.state / "catalog.json", json.dumps(data, indent=2))

    def load_json(self, name: str) -> dict[str, Any]:
        path = self.state / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def save_json(self, name: str, data: dict[str, Any]) -> None:
        self.state.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.state / name, json.dumps(data))

    # --- control requests -------------------------------------------------------------------
    def request(self, action: str, target: str) -> Path:
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r}; use one of {', '.join(ACTIONS)}")
        self.control.mkdir(parents=True, exist_ok=True)
        path = self.control / f"{int(time.time() * 1000)}-{action}.json"
        path.write_text(json.dumps({"action": action, "target": target,
                                    "requested_at": _now()}), encoding="utf-8")
        return path

    def pending_requests(self) -> list[tuple[Path, dict[str, str]]]:
        if not self.control.exists():
            return []
        return [(p, json.loads(p.read_text(encoding="utf-8")))
                for p in sorted(self.control.glob("*.json"))]

    def mark_applied(self, path: Path, outcome: str) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        data.update(outcome=outcome, applied_at=_now())
        (self.control / "applied").mkdir(parents=True, exist_ok=True)
        _atomic_write(self.control / "applied" / path.name, json.dumps(data))
        path.unlink()

    # --- moving files ------------------------------------------------------------------------
    def place(self, src: Path, entry: Entry, *, move: bool) -> str:
        """Put a verified document where its status says: lake/ or archive/."""
        base = self.lake if entry.status == "active" else self.archive
        dest = base / _safe(entry.doc_id) / _safe(entry.version) / entry.file
        _transfer(src, dest, move=move)
        return self.rel(dest)

    def relocate(self, entry: Entry) -> str:
        """Move a known document between lake/ and archive/ after its status changed."""
        current = self.abs(entry.path)
        base = self.lake if entry.status == "active" else self.archive
        dest = base / _safe(entry.doc_id) / _safe(entry.version) / entry.file
        if current.exists() and current.resolve() != dest.resolve():
            _transfer(current, dest, move=True)
            _prune_empty(current.parent, stop=self.root)
        return self.rel(dest) if dest.exists() else entry.path

    def to_quarantine(self, src: Path, run_id: str, reasons: dict[str, Any], *,
                      move: bool) -> str:
        dest = self.quarantine / run_id / src.name
        _transfer(src, dest, move=move)
        (dest.parent / f"{src.name}.reason.json").write_text(
            json.dumps(reasons, indent=2), encoding="utf-8")
        return self.rel(dest)

    def remove(self, entry: Entry) -> None:
        path = self.abs(entry.path)
        if path.exists() and (self.lake in path.parents or self.archive in path.parents):
            path.unlink()
            _prune_empty(path.parent, stop=self.root)

    def write_report(self, kind: str, run_id: str, data: dict[str, Any], text: str) -> Path:
        self.reports.mkdir(parents=True, exist_ok=True)
        path = self.reports / f"{kind}-{run_id}.json"
        _atomic_write(path, json.dumps(data, indent=2))
        _atomic_write(self.reports / f"{kind}-{run_id}.txt", text)
        return path

    def latest_report(self, kind: str, run_id: str | None = None) -> Path | None:
        if run_id:
            path = self.reports / f"{kind}-{run_id}.txt"
            return path if path.exists() else None
        found = sorted(self.reports.glob(f"{kind}-*.txt"))
        return found[-1] if found else None


def new_run_id() -> str:
    """UTC time to the millisecond: sortable, and unique per run."""
    now = time.time()
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime(now)) + f"-{int(now * 1000) % 1000:03d}"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe(part: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "-" for c in part) or "unknown"


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _transfer(src: Path, dest: Path, *, move: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dest.resolve():
        return
    if move:
        shutil.move(str(src), str(dest))
    else:
        shutil.copy2(src, dest)


def _prune_empty(directory: Path, stop: Path) -> None:
    """Remove empty folders left behind by a move, up to (not including) the lake root."""
    stop = stop.resolve()
    current = directory.resolve()
    while current != stop and stop in current.parents:
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


# --- command line -----------------------------------------------------------------------------
def _print_status(lake: DataLake) -> int:
    catalog = lake.load_catalog()
    if not catalog.entries:
        print("the catalog is empty: run the ingestion Job first")
        return 0
    rows = sorted(catalog.entries.values(), key=lambda e: (e.doc_id, e.version))
    width = max(len(e.doc_key) for e in rows)
    print(f"{'document version'.ljust(width)}  status    pinned  origin          path")
    for e in rows:
        print(f"{e.doc_key.ljust(width)}  {e.status.ljust(8)}  {'yes' if e.pinned else 'no ':6}"
              f"  {e.origin[:14].ljust(14)}  {e.path}")
    print(f"\n{sum(e.status == 'active' for e in rows)} active, "
          f"{sum(e.status == 'archived' for e in rows)} archived, "
          f"{len(catalog.deleted)} deleted on request")
    pending = lake.pending_requests()
    if pending:
        print(f"{len(pending)} request(s) waiting for the next ingestion run:")
        for _, req in pending:
            print(f"  {req['action']} {req['target']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pipeline.lake",
                                     description="Inspect the data lake and queue requests.")
    parser.add_argument("--root", default=os.environ.get("DATALAKE_DIR", "/datalake"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="list every document version and where it is")
    req = sub.add_parser("request", help="queue a restore, delete or release for the next run")
    req.add_argument("action", choices=ACTIONS)
    req.add_argument("target", help="doc_key (doc_id@version); release takes a doc_id")
    rep = sub.add_parser("report", help="print a verification or ingestion report")
    rep.add_argument("kind", choices=("verify", "ingest"))
    rep.add_argument("run_id", nargs="?")
    args = parser.parse_args(argv)
    lake = DataLake(args.root)
    if args.command == "status":
        return _print_status(lake)
    if args.command == "request":
        path = lake.request(args.action, args.target)
        print(f"queued: {args.action} {args.target} ({lake.rel(path)}); "
              "it is applied by the next ingestion Job run")
        return 0
    found = lake.latest_report(args.kind, args.run_id)
    if found is None:
        print(f"no {args.kind} report yet")
        return 1
    sys.stdout.write(found.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
