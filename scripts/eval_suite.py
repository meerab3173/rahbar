#!/usr/bin/env python3
# ruff: noqa
"""Part B: the offline test suite at three levels (paper 4.2.5, Table 1).

  module      embedding: paraphrase vs unrelated cosine gap, ms per chunk, determinism
  component   retrieval: recall@3, recall@5, MRR, nDCG@5; noise test with typos in queries
  end-to-end  answers: faithfulness (LLM judge), abstention correctness, citation validity,
              latency p50 / p95

Seeds are fixed (typos: random seed 42; judge: temperature 0, seed 42) and the whole suite
runs --runs times (default 2) so the difference per metric can be reported. The semantic
cache is flushed before every pass, so no answer comes from an earlier pass.

  make suite                 # both runs, about 1 to 2 hours on a laptop
  make suite RUNS=1          # one run, for a quick try
  make judge-agreement FILE=eval/results/suite-<stamp>-judge.csv   # after hand labels

Writes eval/results/suite-<stamp>.json, .md (summary table) and -judge.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from rag_client import ROOT, ask, embed, ollama

RESULTS = ROOT / "eval" / "results"

# ---- module level: 15 paraphrase pairs and 15 unrelated pairs -------------------------
PARAPHRASES = [
    ("What CGPA puts a student on academic warning?", "Below which GPA does a student get a warning?"),
    ("How much attendance is required to sit the final exam?", "What is the minimum class attendance for finals?"),
    ("Can a student repeat a course?", "Is it allowed to retake a subject?"),
    ("Until when can I withdraw from a course?", "What is the last date to drop a course?"),
    ("How many credit hours is a normal semester load?", "What is the usual number of credit hours per term?"),
    ("Can I freeze a semester?", "Is it possible to take a semester off?"),
    ("What is the penalty for plagiarism?", "What happens if I copy someone else's work?"),
    ("How is the CGPA calculated?", "How do they compute the cumulative grade point average?"),
    ("Is the use of AI tools allowed in assignments?", "Can I use ChatGPT for my coursework?"),
    ("What grade is needed to pass a course?", "What is the minimum passing grade?"),
    ("When is a student dismissed from the university?", "Under what conditions is a student expelled?"),
    ("How do I apply for a transcript?", "What is the process to get my transcript?"),
    ("Is there a fee for late registration?", "Do I pay a fine if I register late?"),
    ("What is the summer semester for?", "What courses can be taken in the summer session?"),
    ("What is required to graduate with distinction?", "How can I get a degree with honours?"),
]
UNRELATED = [
    ("What CGPA puts a student on academic warning?", "How do I bake a chocolate cake?"),
    ("How much attendance is required to sit the final exam?", "Who won the football world cup?"),
    ("Can a student repeat a course?", "What is the capital of Australia?"),
    ("Until when can I withdraw from a course?", "How do volcanoes erupt?"),
    ("How many credit hours is a normal semester load?", "Best places to visit in Hunza"),
    ("Can I freeze a semester?", "How do I change a car tyre?"),
    ("What is the penalty for plagiarism?", "Recipe for chicken biryani"),
    ("How is the CGPA calculated?", "Why is the sky blue?"),
    ("Is the use of AI tools allowed in assignments?", "How long do cats live?"),
    ("What grade is needed to pass a course?", "What is the price of gold today?"),
    ("When is a student dismissed from the university?", "How to grow tomatoes on a balcony"),
    ("How do I apply for a transcript?", "Lyrics style of 90s pop music"),
    ("Is there a fee for late registration?", "How far is the moon from the earth?"),
    ("What is the summer semester for?", "Rules of cricket for beginners"),
    ("What is required to graduate with distinction?", "How do airplanes stay in the air?"),
]
CHUNK = ("A student whose CGPA falls below 2.00 at the end of a regular semester is placed on "
         "academic warning. A student on warning may register for a reduced load only, and must "
         "meet the advisor before registration. If the CGPA stays below 2.00 for three consecutive "
         "regular semesters, the student's enrolment is closed. Summer semester results count in "
         "the CGPA but a summer semester is not counted as one of the three semesters. ") * 2


def cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def module_level(a) -> dict:
    def mean_pair(pairs):
        left = embed(a.ollama, a.embed_model, [p[0] for p in pairs])
        right = embed(a.ollama, a.embed_model, [p[1] for p in pairs])
        return sum(cos(x, y) for x, y in zip(left, right)) / len(pairs)
    para, unrel = mean_pair(PARAPHRASES), mean_pair(UNRELATED)
    embed(a.ollama, a.embed_model, ["warm up"])
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        embed(a.ollama, a.embed_model, [CHUNK])
        times.append((time.perf_counter() - t0) * 1000)
    v1 = embed(a.ollama, a.embed_model, [CHUNK])[0]
    v2 = embed(a.ollama, a.embed_model, [CHUNK])[0]
    return {"paraphrase_cosine_mean": round(para, 4), "unrelated_cosine_mean": round(unrel, 4),
            "cosine_gap": round(para - unrel, 4),
            "embed_ms_per_chunk": round(sorted(times)[len(times) // 2], 1),
            "determinism_max_abs_diff": max(abs(x - y) for x, y in zip(v1, v2))}


# ---- component level ------------------------------------------------------------------
def is_rel(r, gt, tol):
    return r.get("source") == gt["doc"] and gt.get("page") is not None \
        and abs(int(r.get("page") or -99) - int(gt["page"])) <= tol


def retrieval_scores(gts, retrieved, tol):
    """recall@k = share of ground-truth (doc, page) items found in the top k;
    MRR = 1 / rank of the first relevant chunk; nDCG@5 with binary relevance, where a
    ground-truth item only counts the first time it is found."""
    gts = [g for g in gts if g.get("page") is not None]
    if not gts:
        return None
    def recall(k):
        return sum(any(is_rel(r, g, tol) for r in retrieved[:k]) for g in gts) / len(gts)
    rr = next((1 / i for i, r in enumerate(retrieved, 1) if any(is_rel(r, g, tol) for g in gts)), 0.0)
    seen, dcg = set(), 0.0
    for i, r in enumerate(retrieved[:5], 1):
        for j, g in enumerate(gts):
            if j not in seen and is_rel(r, g, tol):
                seen.add(j)
                dcg += 1 / math.log2(i + 1)
                break
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(len(gts), 5) + 1))
    return {"recall@3": recall(3), "recall@5": recall(5), "mrr": rr, "ndcg@5": dcg / idcg}


def typo(text, rng):
    words = text.split()
    idx = [i for i, w in enumerate(words) if len(w) > 3 and w.isalpha()]
    pick = rng.sample(idx, max(1, len(idx) // 2)) if idx else []
    for i in pick:
        w = list(words[i])
        j = rng.randrange(1, len(w) - 1)
        w[j], w[j + 1] = w[j + 1], w[j]
        words[i] = "".join(w)
    return " ".join(words)


# ---- end-to-end -----------------------------------------------------------------------
def citation_validity(resp):
    cits = resp.get("citations") or []
    if not cits:
        return None
    got = {(r.get("source"), r.get("page")) for r in resp.get("retrieved", [])}
    return sum((c.get("source"), c.get("page")) in got for c in cits) / len(cits)


def context_for(trace_id) -> str:
    """The exact evidence the model saw, from the D2 trace tool."""
    if not trace_id:
        return ""
    try:
        out = subprocess.run(["make", "-s", "trace", f"ID={trace_id}"], cwd=ROOT,
                             capture_output=True, text=True, timeout=120).stdout
    except Exception:
        return ""
    if "--- user ---" in out:
        out = out.split("--- user ---", 1)[1]
        lines: list[str] = []
        for line in out.splitlines():
            if line.startswith("---") and lines:
                break
            lines.append(line)
        out = "\n".join(lines)
    return out[-6000:]


def judge(a, context, answer) -> str | None:
    if not context.strip():
        return None
    prompt = ("You check answers from a question-answering system. CONTEXT is the only evidence "
              "the system was given. Reply with exactly one word: YES if every factual claim in "
              "ANSWER is supported by CONTEXT, otherwise NO.\n\nCONTEXT:\n" + context +
              "\n\nANSWER:\n" + answer + "\n\nVerdict (YES or NO):")
    try:
        r = ollama(a.ollama, "/api/generate", {"model": a.judge_model, "prompt": prompt,
                   "stream": False, "options": {"temperature": 0, "seed": 42}})
    except Exception as exc:
        print(f"    judge failed: {exc}", file=sys.stderr)
        return None
    return "yes" if r.get("response", "").strip().lower().startswith("yes") else "no"


def pct(values, q):
    v = sorted(values)
    return v[max(0, math.ceil(q / 100 * len(v)) - 1)] if v else None


def mean(values):
    values = [x for x in values if x is not None]
    return round(sum(values) / len(values), 4) if values else None


def flush(a):
    r = subprocess.run(shlex.split(a.flush_cmd), capture_output=True, text=True)
    if r.returncode:
        print(f"  WARNING: cache flush failed ({r.stderr.strip()[:120]}); "
              "cached answers will be counted", file=sys.stderr)


def one_run(a, golden, run_no) -> tuple[dict, list[dict]]:
    print(f"\n=== run {run_no}: module level", file=sys.stderr)
    module = module_level(a)
    flush(a)
    rows = []
    for i, q in enumerate(golden, 1):
        resp, secs = ask(q["question"])
        answered = resp.get("answered_from_context") is True
        row = {"id": q["id"], "question": q["question"], "answerable": q["answerable"],
               "hard": bool(q.get("hard")), "origin": q.get("origin"),
               "trace_id": resp.get("request_id"), "seconds": round(secs, 2),
               "cached": bool(resp.get("cached")), "answered": answered,
               "answer": (resp.get("answer") or resp.get("error") or "").strip(),
               "citation_validity": citation_validity(resp),
               "retrieval": retrieval_scores(q.get("ground_truth", []),
                                             resp.get("retrieved", []), a.page_tolerance)
               if q["answerable"] else None,
               "faithful": None}
        if q["answerable"] and answered:
            row["faithful"] = judge(a, context_for(row["trace_id"]), row["answer"])
        rows.append(row)
        print(f"  [{i}/{len(golden)}] {q['id']:<12} {secs:5.1f}s answered={answered} "
              f"faithful={row['faithful']}", file=sys.stderr)

    print(f"=== run {run_no}: noise test (typos in the queries)", file=sys.stderr)
    flush(a)
    rng, noisy = random.Random(42), []
    for q in ([] if a.no_noise else golden):
        if q["answerable"] and q.get("ground_truth"):
            tq = typo(q["question"], rng)
            resp, _ = ask(tq)
            noisy.append(retrieval_scores(q["ground_truth"], resp.get("retrieved", []),
                                          a.page_tolerance))

    def ret(key, src):
        return mean([r[key] for r in src if r])
    clean = [r["retrieval"] for r in rows if r["retrieval"]]
    lat = [r["seconds"] for r in rows if not r["cached"]]
    unans = [r for r in rows if not r["answerable"]]
    ans = [r for r in rows if r["answerable"]]
    judged = [r["faithful"] for r in rows if r["faithful"] is not None]
    summary = {
        "module": module,
        "component": {k: ret(k, clean) for k in ("recall@3", "recall@5", "mrr", "ndcg@5")},
        "component_typos": {k: ret(k, noisy) for k in ("recall@3", "recall@5", "mrr", "ndcg@5")},
        "end_to_end": {
            "faithfulness": round(judged.count("yes") / len(judged), 4) if judged else None,
            "faithfulness_judged": len(judged),
            "abstention_correctness": round(sum(not r["answered"] for r in unans) / len(unans), 4)
            if unans else None,
            "false_abstention_rate": round(sum(not r["answered"] for r in ans) / len(ans), 4)
            if ans else None,
            "citation_validity": mean([r["citation_validity"] for r in rows]),
            "latency_p50_s": pct(lat, 50), "latency_p95_s": pct(lat, 95),
            "cached_answers": sum(r["cached"] for r in rows),
        },
        "hard_only": {k: ret(k, [r["retrieval"] for r in rows if r["hard"] and r["retrieval"]])
                      for k in ("recall@5", "mrr")},
    }
    return summary, rows


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3g}" if abs(v) < 1e-3 and v else f"{v:.4f}".rstrip("0").rstrip(".")
    return str(v)


def table(runs, golden, a) -> str:
    lines = [f"# RAHBAR Part B suite  {datetime.now():%Y-%m-%d %H:%M}",
             f"golden: {len(golden)} questions ({sum(q['answerable'] for q in golden)} answerable, "
             f"{sum(not q['answerable'] for q in golden)} unanswerable, "
             f"{sum(bool(q.get('hard')) for q in golden)} hard) · embed {a.embed_model} · "
             f"judge {a.judge_model} · page tolerance {a.page_tolerance}", "",
             "| level | metric | " + " | ".join(f"run {i+1}" for i in range(len(runs))) +
             (" | diff |" if len(runs) > 1 else "")]
    lines.append("|---|---|" + "---|" * len(runs) + ("---|" if len(runs) > 1 else ""))
    for level in ("module", "component", "component_typos", "end_to_end", "hard_only"):
        for k in runs[0][level]:
            vals = [r[level][k] for r in runs]
            diff = ""
            if len(runs) > 1:
                d = (vals[1] - vals[0]) if all(isinstance(v, (int, float)) for v in vals) else None
                diff = f" {fmt(d)} |"
            lines.append(f"| {level} | {k} | " + " | ".join(fmt(v) for v in vals) + " |" + diff)
    c, t = runs[0]["component"], runs[0]["component_typos"]
    if any(v is not None for v in t.values()):
        lines += ["", "Noise test (run 1, typos minus clean): " + ", ".join(
            f"{k} {fmt((t[k] or 0) - (c[k] or 0))}" for k in c)]
    else:
        lines += ["", "Noise test skipped (--no-noise)."]
    return "\n".join(lines)


def agreement(path: Path) -> int:
    rows = list(csv.DictReader(path.open()))
    lab = [r for r in rows if r.get("human", "").strip().lower() in ("yes", "no")]
    if not lab:
        print("no rows labelled yet: put yes or no in the 'human' column")
        return 1
    agree = sum(r["human"].strip().lower() == r["judge"].strip().lower() for r in lab)
    line = f"Judge agreement with hand labels: {agree}/{len(lab)} = {agree/len(lab):.0%}"
    print(line)
    md = Path(str(path).replace("-judge.csv", ".md"))
    if md.exists():
        md.write_text(md.read_text().rstrip() + "\n\n" + line + "\n")
        print(f"added to {md.relative_to(ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=str(ROOT / "eval" / "golden.jsonl"))
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--embed-model", default="nomic-embed-text")
    ap.add_argument("--judge-model", default="llama3.2:3b")
    ap.add_argument("--page-tolerance", type=int, default=0)
    ap.add_argument("--label", default="", help="added to the file name, e.g. baseline")
    ap.add_argument("--flush-cmd", default="kubectl -n rahbar exec deploy/redis -- redis-cli FLUSHALL")
    ap.add_argument("--no-noise", action="store_true", help="skip the typo pass (faster)")
    ap.add_argument("--agreement", help="a -judge.csv with the 'human' column filled in")
    a = ap.parse_args()
    if a.agreement:
        return agreement(Path(a.agreement))

    random.seed(42)
    golden = [json.loads(x) for x in Path(a.golden).read_text().splitlines() if x.strip()]
    runs, all_rows = [], []
    for n in range(1, a.runs + 1):
        s, rows = one_run(a, golden, n)
        runs.append(s)
        all_rows.append(rows)

    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = RESULTS / f"suite-{a.label + '-' if a.label else ''}{stamp}"
    md = table(runs, golden, a)
    Path(f"{base}.json").write_text(json.dumps({"summary": runs, "rows": all_rows}, indent=2))
    Path(f"{base}.md").write_text(md + "\n")
    with open(f"{base}-judge.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "trace_id", "question", "answer", "judge", "human"])
        for r in all_rows[0]:
            if r["faithful"] is not None:
                w.writerow([r["id"], r["trace_id"], r["question"], r["answer"][:300].replace("\n", " "), r["faithful"], ""])
    print(md)
    print(f"\nwrote {base.relative_to(ROOT)}.json, .md and -judge.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
