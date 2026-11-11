#!/usr/bin/env python3
"""Choose CACHE_THRESHOLD with data (section 3.3).

Embeds every pair in eval/cache_pairs.yaml with the pinned query embedding, then, for
each candidate threshold, counts true hits (paraphrases served from cache) and false hits
(different questions that would be served the wrong answer).

    python eval/cache_threshold.py                          # Ollama on localhost
    python eval/cache_threshold.py --embed-model hash-bow-768

Ship the lowest threshold with zero false hits, and paste the table into PIPELINE.md.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))

from app.embedding import cosine, get_embedder  # noqa: E402
from app.services.guards import exact_signature  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pairs", type=Path, default=ROOT / "eval" / "cache_pairs.yaml")
    ap.add_argument("--embed-model", default=os.environ.get("EMBED_MODEL", "nomic-embed-text"))
    ap.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    ap.add_argument("--thresholds", default="0.85,0.88,0.90,0.92,0.94,0.95,0.96,0.97,0.98")
    args = ap.parse_args()

    pairs = yaml.safe_load(args.pairs.read_text())["pairs"]
    embedder = get_embedder(args.embed_model, args.ollama_url)
    scored = [(p["same"], cosine(embedder.embed(p["a"]), embedder.embed(p["b"])), p)
              for p in pairs]

    print(f"\nEmbedding model: {args.embed_model}\n")
    print("| Similarity | Same answer? | Question A | Question B |")
    print("| --- | --- | --- | --- |")
    for same, s, p in sorted(scored, key=lambda t: -t[1]):
        print(f"| {s:.4f} | {'yes' if same else 'NO'} | {p['a']} | {p['b']} |")

    n_same = sum(1 for same, _, _ in scored if same)
    for guard in (False, True):
        label = "with" if guard else "without"
        print(f"\n{label.capitalize()} the exact-term guard (a hit also needs the same "
              "numbers, clause numbers, grades and codes in both questions):\n")
        print("| Threshold | True hits | False hits | Verdict |")
        print("| --- | --- | --- | --- |")
        chosen = None
        for t in (float(x) for x in args.thresholds.split(",")):
            def hit(score: float, p: dict) -> bool:
                return score >= t and (not guard or
                                       exact_signature(p["a"]) == exact_signature(p["b"]))
            true_hits = sum(1 for same, s, p in scored if same and hit(s, p))
            false = [p for same, s, p in scored if not same and hit(s, p)]
            verdict = "ruled out" if false else "safe"
            if not false and chosen is None:
                chosen = t
            example = f" (false hit: \"{false[0]['a']}\" ~ \"{false[0]['b']}\")" \
                if false else ""
            print(f"| {t:.2f} | {true_hits}/{n_same} | {len(false)} | {verdict}{example} |")
        print(f"\nLowest threshold with zero false hits ({label} guard): {chosen}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
