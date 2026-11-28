#!/usr/bin/env python3
"""Measure CPU and memory per container while a command runs (Assignment 03, F5).

    python scripts/measure_resources.py --label ingest -- make ingest
    python scripts/measure_resources.py --label idle --seconds 60

Samples `kubectl top pods --containers` every few seconds (metrics-server refreshes about
every 15 s), keeps every sample in eval/results/resources/<label>-<time>.tsv, and prints
the peak and the median per container. Requests and limits in k8s/ are set from these
numbers: request ~ the median under load, limit ~ 1.5 x the peak (docs/RESOURCES.md).
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "eval" / "results" / "resources"


def sample(namespace: str) -> list[tuple[str, str, int, int]]:
    """(pod, container, cpu millicores, memory MiB) for every running container."""
    try:
        out = subprocess.run(["kubectl", "-n", namespace, "top", "pods", "--containers",
                              "--no-headers"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows = []
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        pod, container, cpu, mem = parts[:4]
        try:
            millicores = int(cpu.rstrip("m")) if cpu.endswith("m") else int(float(cpu) * 1000)
            mib = int(mem.rstrip("Mi")) if mem.endswith("Mi") else \
                int(float(mem.rstrip("Gi")) * 1024) if mem.endswith("Gi") else \
                int(int(mem.rstrip("Ki")) / 1024)
        except ValueError:
            continue
        rows.append((pod, container, millicores, mib))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--label", required=True, help="what is being measured, e.g. ingest")
    parser.add_argument("--namespace", default="rahbar")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--seconds", type=float, default=0, help="measure for this long "
                        "instead of running a command")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- then the command")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command and not args.seconds:
        parser.error("give a command after --, or --seconds")

    samples: list[tuple[float, str, str, int, int]] = []
    done = threading.Event()

    def poll() -> None:
        while not done.is_set():
            now = time.time()
            samples.extend((now, *row) for row in sample(args.namespace))
            done.wait(args.interval)

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    code = 0
    if command:
        code = subprocess.call(command)
    else:
        time.sleep(args.seconds)
    time.sleep(args.interval)          # one more sample after the command ends
    done.set()
    thread.join()

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    raw = OUT / f"{args.label}-{stamp}.tsv"
    raw.write_text("epoch\tpod\tcontainer\tcpu_m\tmemory_mi\n" + "".join(
        f"{t:.0f}\t{p}\t{c}\t{cpu}\t{mem}\n" for t, p, c, cpu, mem in samples))

    by_container: dict[str, list[tuple[int, int]]] = {}
    for _, _, container, cpu, mem in samples:
        by_container.setdefault(container, []).append((cpu, mem))
    lines = [f"RESOURCES  {args.label}  ({len(samples)} samples, every {args.interval:.0f} s)",
             f"{'container':<14} {'cpu median':>10} {'cpu peak':>9} {'mem median':>11} "
             f"{'mem peak':>9}"]
    for container, values in sorted(by_container.items()):
        cpus = [v[0] for v in values]
        mems = [v[1] for v in values]
        lines.append(f"{container:<14} {int(statistics.median(cpus)):>9}m {max(cpus):>8}m "
                     f"{int(statistics.median(mems)):>8}Mi {max(mems):>7}Mi")
    summary = "\n".join(lines) + "\n"
    (OUT / f"{args.label}-{stamp}.txt").write_text(summary)
    print(summary + f"raw samples: {raw.relative_to(OUT.parents[2])}")
    return code


if __name__ == "__main__":
    sys.exit(main())
