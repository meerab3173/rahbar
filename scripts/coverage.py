#!/usr/bin/env python3
# ruff: noqa
"""E2 / E4: query coverage. For every live query, the most similar golden question
(cosine of nomic-embed-text embeddings, the same model the pipeline retrieves with).

  .venv/bin/python scripts/coverage.py                       # threshold 0.75, need 85 %
  .venv/bin/python scripts/coverage.py --threshold 0.8 --min-share 0.85
  .venv/bin/python scripts/coverage.py --extra eval/questions/coverage-gap.jsonl   # E4

Exits 1 when the share of covered queries is below --min-share (the paper's 85 % example).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from rag_client import ROOT, embed


def load(paths, key):
    out = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                out.append((row.get("id", "?"), row[key], row.get("in_domain")))
    return out


def cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / (
        math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument("--min-share", type=float, default=0.85)
    ap.add_argument("--extra", action="append", default=[], help="extra golden jsonl (E4)")
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--model", default="nomic-embed-text")
    a = ap.parse_args()

    live = load(sorted((ROOT / "eval" / "live").glob("queries-*.jsonl")), "query")
    golden = load([ROOT / "eval" / "golden.jsonl", *a.extra], "question")
    lv = embed(a.ollama, a.model, [q for _, q, _ in live])
    gv = embed(a.ollama, a.model, [q for _, q, _ in golden])

    covered, uncovered = 0, []
    print(f"COVERAGE  {len(live)} live queries vs {len(golden)} golden questions  "
          f"(threshold {a.threshold})")
    for (lid, q, dom), v in zip(live, lv):
        score, best = max((cos(v, g), gq) for g, (_, gq, _) in zip(gv, golden))
        ok = score >= a.threshold
        covered += ok
        if not ok:
            uncovered.append((score, lid, q, dom))
        print(f"  {'ok ' if ok else 'GAP'} {score:.3f}  {lid:<16} {q[:55]:<55} -> {best[:45]}")
    share = covered / len(live)
    print(f"\ncovered {covered}/{len(live)} = {share:.0%}   (need {a.min_share:.0%})")
    print("\nUNCOVERED (group these into topics in the report):")
    for score, lid, q, dom in sorted(uncovered):
        print(f"  {score:.3f}  {lid:<16} in_domain={dom}  {q}")
    if share < a.min_share:
        print(f"\nFAIL: coverage {share:.0%} is below {a.min_share:.0%}")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
