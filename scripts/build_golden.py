#!/usr/bin/env python3
"""B1: build eval/golden.jsonl.

Sources
  eval/golden.yaml                    21 Assignment 02 questions  -> origin "synthetic"
  eval/questions/extra-synthetic.yaml  4 extra questions          -> origin "synthetic"
  eval/questions/human-*.jsonl        hand-written questions      -> origin "human" (copied)

The YAML questions name a (source, section). Part B wants a document AND a page, so each
one is asked once through `make ask` and the page of the first retrieved chunk from that
source and section becomes the ground-truth page. A person then checks it in the PDF and
marks the question reviewed:

  .venv/bin/python scripts/build_golden.py                       # build
  .venv/bin/python scripts/build_golden.py --mark-reviewed q01,q02 --by meerab
  .venv/bin/python scripts/build_golden.py --check               # counts against Part B
"""
from __future__ import annotations

import argparse
import json
import sys

import yaml

from rag_client import ROOT, ask, section_matches

OUT = ROOT / "eval" / "golden.jsonl"
HARD_TYPES = {"version", "paraphrase", "multi", "batch-b"}


def from_yaml(path, author):
    data = yaml.safe_load(path.read_text())
    rows = []
    for q in data.get("questions", []):
        abstain = bool(q.get("must_abstain", False))
        row = {"id": q["id"], "question": q["question"], "answerable": not abstain,
               "hard": bool(q.get("hard", q.get("type") in HARD_TYPES)),
               "origin": "synthetic", "author": author, "reviewed_by": "",
               "type": q.get("type", "ordinary"), "ground_truth": []}
        if q.get("expect"):
            row["expect"] = q["expect"]
        if not abstain and q.get("relevant"):
            resp, _ = ask(q["question"])
            retrieved = resp.get("retrieved", [])
            for rel in q["relevant"]:
                hit = next((r for r in retrieved if r.get("source") == rel["source"]
                            and section_matches(rel.get("section", ""), r.get("section", ""))),
                           None)
                gt = {"doc": rel["source"], "page": hit["page"] if hit else None}
                if gt not in row["ground_truth"]:
                    row["ground_truth"].append(gt)
            row["_answer"] = (resp.get("answer") or resp.get("error") or "")[:110]
        print(f"  {row['id']:<6} {'answerable' if row['answerable'] else 'abstain':<10} "
              f"{row['ground_truth']}", file=sys.stderr)
        rows.append(row)
    return rows


def build():
    rows = []
    for path in sorted((ROOT / "eval" / "questions").glob("human-*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                q = json.loads(line)
                q.setdefault("type", "human")
                rows.append(q)
    print("asking the YAML questions once each to find their pages ...", file=sys.stderr)
    rows += from_yaml(ROOT / "eval" / "golden.yaml", "group-a2")
    extra = ROOT / "eval" / "questions" / "extra-synthetic.yaml"
    if extra.exists():
        rows += from_yaml(extra, "meerab")

    print("\nCHECK THESE PAGES IN THE PDF (id | doc p.page | what RAHBAR answered):")
    for r in rows:
        if r["origin"] == "synthetic" and r["answerable"]:
            pages = ", ".join(f"{g['doc']} p{g['page']}" for g in r["ground_truth"]) or "-"
            flag = "   <-- NO PAGE FOUND, fill in by hand" if any(
                g["page"] is None for g in r["ground_truth"]) else ""
            print(f"  {r['id']:<6} | {pages} | {r.get('_answer', '')}{flag}")
    with OUT.open("w") as f:
        for r in rows:
            r.pop("_answer", None)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(rows)} questions to {OUT.relative_to(ROOT)}")
    check()


def load():
    return [json.loads(x) for x in OUT.read_text().splitlines() if x.strip()]


def mark(ids, by):
    rows, wanted = load(), set(ids)
    for r in rows:
        if r["id"] in wanted:
            r["reviewed_by"] = by
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    missing = wanted - {r["id"] for r in rows}
    print(f"marked {len(wanted - missing)} reviewed by {by}" +
          (f"; unknown ids: {sorted(missing)}" if missing else ""))
    check()


def check() -> int:
    rows = load()
    n = {
        "total": (len(rows), 25),
        "answerable": (sum(r["answerable"] for r in rows), 15),
        "unanswerable": (sum(not r["answerable"] for r in rows), 5),
        "hard": (sum(bool(r.get("hard")) for r in rows), 5),
        "human": (sum(r["origin"] == "human" for r in rows), 8),
        "synthetic reviewed": (sum(r["origin"] == "synthetic" and bool(r.get("reviewed_by"))
                                   for r in rows), 8),
        "answerable with a page": (sum(r["answerable"] and bool(r["ground_truth"]) and all(
            g.get("page") for g in r["ground_truth"]) for r in rows),
            sum(r["answerable"] for r in rows)),
    }
    print("\nGOLDEN SET against Part B")
    ok = True
    for k, (have, need) in n.items():
        good = have >= need
        ok &= good
        print(f"  {k:<24} {have:>3}  (need {need})  {'OK' if good else 'MISSING'}")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mark-reviewed", help="comma-separated ids you checked in the PDF")
    ap.add_argument("--by", default="meerab")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.mark_reviewed:
        mark([x.strip() for x in a.mark_reviewed.split(",") if x.strip()], a.by)
    elif a.check:
        sys.exit(check())
    else:
        build()
