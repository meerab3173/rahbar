"""Read the trace records (Assignment 03, D2 and D3).

    python -m app.tracetool show <trace_id>          # D2: the exact context and prompt
    python -m app.tracetool list [-n 20]             # the latest traces, one line each
    python -m app.tracetool summary [--since 1h]     # D3: metrics over a time window
    python -m app.tracetool summary --since 2h --json

Inside the cluster (the api pods mount the traces volume at /traces):
    kubectl -n rahbar exec deploy/api -- python -m app.tracetool show <trace_id>
On a copy in the repository (make traces-pull):
    python api/app/tracetool.py --dir traces show <trace_id>
"""

from __future__ import annotations

import argparse
import calendar
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_DIR = os.environ.get("TRACE_DIR") or "/traces"


def load(directory: str) -> list[dict[str, Any]]:
    traces = []
    for path in sorted(Path(directory).glob("traces-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    traces.sort(key=lambda t: t.get("timestamp", ""))
    return traces


def find(directory: str, trace_id: str) -> dict[str, Any] | None:
    matches = [t for t in load(directory) if str(t.get("trace_id", "")).startswith(trace_id)]
    return matches[-1] if matches else None


def prompt(directory: str, digest: str) -> list[dict[str, str]] | None:
    path = Path(directory) / "prompts" / f"{digest}.json"
    if not path.exists():
        return None
    return list(json.loads(path.read_text(encoding="utf-8"))["messages"])


# --- show ---------------------------------------------------------------------------------------
def show(directory: str, trace_id: str) -> int:
    t = find(directory, trace_id)
    if t is None:
        print(f"no trace {trace_id!r} in {directory}")
        return 1
    out = print
    out(f"TRACE {t['trace_id']}   {t.get('timestamp')}   pod {t.get('pod')}")
    out(f"query            {t.get('query')!r}")
    out(f"rewritten query  {t.get('rewritten_query')!r}")
    emb, col = t.get("embedding") or {}, t.get("collection") or {}
    out(f"embedding        {emb.get('model')}  version {emb.get('version')}")
    out(f"collection       {col.get('name')}  generation {col.get('generation')}"
        f"  (index built with {col.get('index_embed_model')}, {col.get('dimension')}-d, "
        f"chunks {col.get('chunk_size')}/{col.get('chunk_overlap')})")
    out(f"filters          {json.dumps(t.get('filters'))}")
    out(f"model            {json.dumps(t.get('model'))}")
    cache = t.get("cache") or {}
    out(f"cache            {'HIT' if cache.get('hit') else 'miss'}"
        f"{'  similarity ' + str(cache.get('similarity')) if 'similarity' in cache else ''}"
        + (f"  answer first produced by {cache.get('source_trace_id')}"
           if cache.get("hit") else ""))
    out("")
    out("retrieved (the order given to the model, after the rerank):")
    for r in t.get("retrieved") or []:
        out(f"  {str(r.get('rank', '-')):>2}. {r.get('chunk_id')}  doc {r.get('doc_id')}"
            f"@{r.get('doc_version')}  score {r.get('score')}  rank {r.get('rank_score')}"
            f"{'  SUPERSEDED' if r.get('superseded') else ''}  p.{r.get('page')}")
    if t.get("candidates"):
        out(f"vector-search order before the rerank: "
            f"{', '.join(c['chunk_id'] for c in t['candidates'][:10])}")
    out("")
    out("guardrails:")
    for g in t.get("guardrails") or []:
        detail = {k: v for k, v in g.items() if k not in ("rail", "action")}
        out(f"  {g['rail']:<18} {g['action']:<9} {json.dumps(detail) if detail else ''}")
    out(f"latency (ms)     {json.dumps(t.get('latency_ms'))}")
    out("")
    resp = t.get("response") or {}
    out(f"response         {resp.get('answer')!r}")
    out(f"answered from context: {resp.get('answered_from_context')}   "
        f"outcome: {json.dumps(t.get('outcome'))}")
    for c in t.get("citations") or []:
        out(f"  cites {c.get('source')} / {c.get('section')} p.{c.get('page')}")
    p = t.get("prompt") or {}
    if not p:
        out("\n(no model call: the answer came from the cache or a guardrail before generation)")
        return 0
    for n, digest in enumerate(p.get("attempts") or [p.get("sha256")], start=1):
        messages = prompt(directory, digest)
        out("")
        out("=" * 100)
        out(f"PROMPT, attempt {n}   sha256 {digest}")
        out("=" * 100)
        if messages is None:
            out("(prompt file missing)")
            continue
        for m in messages:
            out(f"--- {m['role']} ---")
            out(m["content"])
    return 0


# --- list ---------------------------------------------------------------------------------------
def listing(directory: str, n: int) -> int:
    for t in load(directory)[-n:]:
        resp = t.get("response") or {}
        flags = [g["rail"] for g in t.get("guardrails") or []
                 if g["action"] in ("flagged", "blocked", "abstain", "dropped")]
        out = (f"{t.get('timestamp', '')[:19]}  {t.get('trace_id', '')[:32]:<32}  "
               f"{'HIT ' if (t.get('cache') or {}).get('hit') else '    '}"
               f"{'ABSTAIN ' if not resp.get('answered_from_context') else 'answer  '}"
               f"{','.join(flags):<28} {str(t.get('query'))[:60]}")
        print(out)
    return 0


# --- summary ------------------------------------------------------------------------------------
def parse_since(text: str) -> float:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([smhd])", text.strip())
    if not match:
        raise ValueError(f"--since wants 30m, 2h or 1d, not {text!r}")
    value, unit = float(match.group(1)), match.group(2)
    return value * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]


def epoch(iso: str) -> float:
    """Seconds since 1970 for a trace's UTC timestamp."""
    return float(calendar.timegm(time.strptime(iso[:19], "%Y-%m-%dT%H:%M:%S")))


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo), 4)


def summarise(traces: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(traces)
    abstained = sum(1 for t in traces if not (t.get("response") or {}).get("answered_from_context"))
    lookups = [t for t in traces if "similarity" in (t.get("cache") or {})]
    hits = sum(1 for t in lookups if t["cache"].get("hit"))
    top1 = [float(t["retrieved"][0]["score"]) for t in traces
            if t.get("retrieved") and not (t.get("cache") or {}).get("hit")]
    stages: dict[str, list[float]] = {}
    for t in traces:
        for name, ms in (t.get("latency_ms") or {}).items():
            stages.setdefault(name, []).append(float(ms))
    flagged = sum(1 for t in traces for g in t.get("guardrails") or []
                  if g["rail"] == "injection_scan" and g["action"] == "flagged")
    blocked = sum(1 for t in traces for g in t.get("guardrails") or []
                  if g["rail"] == "injection_echo" and g["action"] == "blocked")
    buckets = [0.5, 0.6, 0.69, 0.75, 0.8, 0.85, 0.9, 1.01]
    histogram, low = {}, 0.0
    for edge in buckets:
        label = f"[{low:.2f}, {min(edge, 1.0):.2f}{']' if edge > 1 else ')'}"
        histogram[label] = sum(1 for s in top1 if low <= s < edge)
        low = edge
    return {
        "queries": n,
        "abstention_rate": round(abstained / n, 4) if n else None,
        "cache_hit_rate": round(hits / len(lookups), 4) if lookups else None,
        "cache_lookups": len(lookups),
        "top1_similarity": {"n": len(top1), "min": percentile(top1, 0), "p10": percentile(top1, .1),
                            "p50": percentile(top1, .5), "p90": percentile(top1, .9),
                            "max": percentile(top1, 1), "histogram": histogram},
        "latency_ms": {name: {"p50": percentile(v, .5), "p95": percentile(v, .95), "n": len(v)}
                       for name, v in sorted(stages.items())},
        "injection_flagged_queries": flagged,
        "injection_blocked_answers": blocked,
    }


def summary(directory: str, since: str | None, until: str | None, as_json: bool) -> int:
    traces = load(directory)
    now = time.time()
    if since:
        start = now - parse_since(since)
        traces = [t for t in traces if epoch(t.get("timestamp", "1970-01-01T00:00:00")) >= start]
    if until:
        end = now - parse_since(until)
        traces = [t for t in traces if epoch(t.get("timestamp", "1970-01-01T00:00:00")) <= end]
    s = summarise(traces)
    window = (f"last {since}" if since else "all time") + (f", up to {until} ago" if until else "")
    s["window"] = window
    if as_json:
        print(json.dumps(s, indent=2))
        return 0
    print(f"TRACE SUMMARY ({window}): {s['queries']} queries")
    if not traces:
        return 0
    print(f"abstention rate      {s['abstention_rate']:.1%}")
    print("cache hit rate       " + (f"{s['cache_hit_rate']:.1%} of {s['cache_lookups']} lookups"
                                     if s["cache_hit_rate"] is not None else "no lookups"))
    t1 = s["top1_similarity"]
    print(f"top-1 similarity     n={t1['n']}  min {t1['min']}  p10 {t1['p10']}  "
          f"median {t1['p50']}  p90 {t1['p90']}  max {t1['max']}")
    for label, count in t1["histogram"].items():
        print(f"    {label:<14} {'#' * count} {count}")
    print("latency per stage (ms)        p50        p95     n")
    for name, v in s["latency_ms"].items():
        print(f"    {name:<22} {v['p50']!s:>10} {v['p95']!s:>10} {v['n']:>5}")
    print(f"injection attempts flagged   {s['injection_flagged_queries']} queries "
          f"({s['injection_blocked_answers']} answers blocked for echoing one)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.tracetool",
                                     description=__doc__.split("\n")[0])
    parser.add_argument("--dir", default=DEFAULT_DIR, help="folder with traces-*.jsonl")
    sub = parser.add_subparsers(dest="command", required=True)
    sh = sub.add_parser("show", help="print one trace, with the exact prompt")
    sh.add_argument("trace_id", help="a trace id, or its first characters")
    ls = sub.add_parser("list", help="the latest traces")
    ls.add_argument("-n", type=int, default=20)
    sm = sub.add_parser("summary", help="metrics over a time window")
    sm.add_argument("--since", help="e.g. 30m, 2h, 1d (default: all)")
    sm.add_argument("--until", help="ignore the most recent span, e.g. 10m")
    sm.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "show":
        return show(args.dir, args.trace_id)
    if args.command == "list":
        return listing(args.dir, args.n)
    return summary(args.dir, args.since, args.until, args.json)


if __name__ == "__main__":
    sys.exit(main())
