#!/usr/bin/env python3
"""Measure recall@k for several (chunk size, overlap) settings, offline (section 3.1).

It runs the indexer's real extract -> clean -> chunk -> embed stages in-process, embeds
the golden questions with the same model, applies the API's superseded filter and its
exact-term rerank, and scores recall@3/5 with run_eval.py's own matching. No cluster and
no Chroma are needed; only the embedding model (Ollama on your laptop, or hash-bow-768).

    # your corpus, with Ollama running and nomic-embed-text pulled
    python eval/tune_chunking.py --corpus corpus/raw --golden eval/golden.yaml \\
        --configs 200:30,400:50,800:100

    # the CI fixtures, fully offline
    python eval/tune_chunking.py --corpus eval/fixtures \\
        --golden eval/fixtures/golden.fixture.yaml --embed-model hash-bow-768

Paste the printed Markdown table into docs/PIPELINE.md. `--json out.json` keeps every
number, and `--floor-report` prints the top similarity per question, which is how
RELEVANCE_FLOOR is chosen (the highest floor that still keeps every answerable question).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "indexer"), str(ROOT / "api"), str(ROOT / "eval")]

from app.retrieval.chroma import exact_boost  # noqa: E402
from app.services.guards import wants_old_version  # noqa: E402
from pipeline.chunk import amplification, chunk  # noqa: E402
from pipeline.clean import clean  # noqa: E402
from pipeline.config import IndexerSettings, get_settings  # noqa: E402
from pipeline.embed import embed  # noqa: E402
from pipeline.extract import extract  # noqa: E402
from pipeline.run import discover  # noqa: E402
from run_eval import load_golden, recall_at_k  # noqa: E402

QUERY_PREFIXES = {"nomic-embed-text": "search_query: "}


def cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default=os.environ.get("CORPUS_DIR", "corpus/raw"))
    ap.add_argument("--golden", type=Path, default=ROOT / "eval" / "golden.yaml")
    ap.add_argument("--configs", default="200:30,400:50,800:100",
                    help="comma-separated size:overlap pairs (tokens = words)")
    ap.add_argument("--embed-model", default=os.environ.get("EMBED_MODEL", "nomic-embed-text"))
    ap.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    ap.add_argument("--hybrid-weight", type=float, default=float(os.environ.get("HYBRID_WEIGHT", 0.15)))
    ap.add_argument("--candidates", type=int, default=4)
    ap.add_argument("--strategy", default="structure", choices=["structure", "fixed"])
    ap.add_argument("--no-ocr", action="store_true", help="index scans without OCR (stage A)")
    ap.add_argument("--no-filter", action="store_true", help="drop the superseded filter (stage E)")
    ap.add_argument("--floor-report", action="store_true")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    base = get_settings()
    questions = load_golden(args.golden)
    docs = []
    for path, version, superseded in discover(args.corpus):
        if path.exists():
            docs.append(extract(str(path), version, superseded, ocr=not args.no_ocr))
    docs = [clean(d) for d in docs]
    if not docs:
        sys.exit(f"no documents in {args.corpus}")

    def settings(size: int = 400, overlap: int = 50) -> IndexerSettings:
        return IndexerSettings(**{**base.__dict__, "embed_model": args.embed_model,
                                  "ollama_url": args.ollama_url, "chunk_size": size,
                                  "chunk_overlap": overlap})  # type: ignore[arg-type]

    prefix = QUERY_PREFIXES.get(args.embed_model.split(":")[0], "")
    # Questions are embedded through the indexer's embed() with the query prefix swapped in.
    from pipeline import embed as embed_mod
    saved = dict(embed_mod.DOCUMENT_PREFIXES)
    embed_mod.DOCUMENT_PREFIXES.clear()
    qvecs = embed([prefix + q["question"] for q in questions], settings())
    embed_mod.DOCUMENT_PREFIXES.update(saved)

    table, report = [], {}
    for pair in args.configs.split(","):
        size, overlap = (int(x) for x in pair.split(":"))
        chunks = [c for d in docs for c in chunk(d, size, overlap, strategy=args.strategy)]
        vectors = embed([c.text for c in chunks], settings(size, overlap))
        amp = amplification(chunks, docs)
        r3s, r5s, by_type, floors = [], [], {}, []
        for q, qv in zip(questions, qvecs):
            old = wants_old_version(q["question"]) or args.no_filter
            pool = [(cos(qv, v), c) for c, v in zip(chunks, vectors) if old or not c.superseded]
            pool.sort(key=lambda t: t[0], reverse=True)
            pool = pool[:5 * args.candidates]
            ranked = sorted(pool, key=lambda t: t[0] + args.hybrid_weight * exact_boost(
                q["question"], t[1].section + " " + t[1].text), reverse=True)
            retrieved = [{"source": c.source, "section": c.section, "score": round(s, 4)}
                         for s, c in ranked[:5]]
            floors.append((q["id"], q.get("type", "ordinary"), bool(q.get("must_abstain")),
                           retrieved[0]["score"] if retrieved else 0.0))
            r3 = recall_at_k(q.get("relevant") or [], retrieved, 3)
            r5 = recall_at_k(q.get("relevant") or [], retrieved, 5)
            if r5 is not None and r3 is not None:
                r3s.append(r3)
                r5s.append(r5)
                by_type.setdefault(q.get("type", "ordinary"), []).append(r5)
        row = {"size": size, "overlap": overlap, "chunks": len(chunks),
               "amplification": amp,
               "recall@3": round(sum(r3s) / len(r3s), 3) if r3s else None,
               "recall@5": round(sum(r5s) / len(r5s), 3) if r5s else None,
               "by_type": {t: round(sum(v) / len(v), 3) for t, v in sorted(by_type.items())},
               "top_scores": floors}
        table.append(row)
        report[pair] = row

    print(f"\nEmbedding model: {args.embed_model}   strategy: {args.strategy}   "
          f"hybrid_weight: {args.hybrid_weight}   corpus: {args.corpus} ({len(docs)} documents)\n")
    print("| Size (tokens) | Overlap (tokens) | recall@3 | recall@5 | Chunks stored | "
          "Amplification |")
    print("| --- | --- | --- | --- | --- | --- |")
    for r in table:
        print(f"| {r['size']} | {r['overlap']} | {r['recall@3']} | {r['recall@5']} | "
              f"{r['chunks']} | {r['amplification']} |")
    types = sorted({t for r in table for t in r["by_type"]})
    print("\n| Size:overlap | " + " | ".join(f"recall@5 {t}" for t in types) + " |")
    print("| --- |" + " --- |" * len(types))
    for r in table:
        print(f"| {r['size']}:{r['overlap']} | "
              + " | ".join(str(r["by_type"].get(t, "-")) for t in types) + " |")
    if args.floor_report:
        print("\nTop similarity per question (choose RELEVANCE_FLOOR between the highest "
              "must-abstain score and the lowest answerable score):")
        for qid, qtype, abstain, score in table[-1]["top_scores"]:
            print(f"  {qid:<5} {qtype:<10} {'ABSTAIN' if abstain else '':<8} {score:.4f}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
