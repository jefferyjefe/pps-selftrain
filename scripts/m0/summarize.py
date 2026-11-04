"""Summarize milestone 0 eval JSONL files per (suite, task, offset_tag).

Usage
  python3 scripts/m0/summarize.py /workspace/runs/m0/exp1_screen/*.jsonl [--json out.json] [--sort rate]
Prints success with a Wilson 95 percent interval, mean and median wall time per episode, infer time and peak memory.
"""

import argparse
import collections
import json
import math
import statistics


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--json", default=None)
    ap.add_argument("--sort", default="task", choices=["task", "rate"])
    args = ap.parse_args()
    groups = collections.defaultdict(list)
    for f in args.files:
        for line in open(f):
            r = json.loads(line)
            groups[(r["suite"], r["task_id"], r.get("offset_tag", ""))].append(r)
    rows = []
    for (suite, tid, tag), rs in groups.items():
        k, n = sum(r["success"] for r in rs), len(rs)
        lo, hi = wilson(k, n)
        walls = [r["wall_s"] for r in rs if r["infer_ms_first"] is not None and r["infer_ms_first"] < 5000]
        inf = [r["infer_ms_mean"] for r in rs if r["infer_ms_mean"] is not None]
        rows.append({
            "suite": suite, "task_id": tid, "task": rs[0]["task"], "prompt": rs[0]["prompt"], "offset_tag": tag,
            "offset": rs[0].get("offset"), "successes": k, "episodes": n, "rate": k / n, "ci95": [lo, hi],
            "episodes_list": sorted(r["episode"] for r in rs),
            "wall_s_mean": statistics.mean(walls) if walls else None,
            "wall_s_median": statistics.median(walls) if walls else None,
            "wall_s_success_mean": statistics.mean([r["wall_s"] for r in rs if r["success"]] or [float("nan")]),
            "infer_ms_mean": statistics.mean(inf) if inf else None,
            "peak_mem_allocated_gb": max(r["peak_mem_allocated_gb"] for r in rs),
            "peak_mem_reserved_gb": max(r["peak_mem_reserved_gb"] for r in rs),
            "num_denoise_steps": rs[0]["num_denoise_steps"], "checkpoint": rs[0]["checkpoint"],
            "errors": sum(r["error"] is not None for r in rs),
        })
    key = (lambda r: (r["suite"], r["offset_tag"], r["task_id"])) if args.sort == "task" else (lambda r: (-r["rate"], r["task_id"]))
    rows.sort(key=key)
    for r in rows:
        print(f"{r['suite']:15s} t{r['task_id']:<3d} {r['offset_tag']:10s} {r['successes']:3d}/{r['episodes']:<3d} "
              f"{100*r['rate']:5.1f}% [{100*r['ci95'][0]:4.1f},{100*r['ci95'][1]:5.1f}] "
              f"wall {r['wall_s_mean'] or float('nan'):5.1f}s infer {r['infer_ms_mean'] or float('nan'):5.1f}ms "
              f"mem {r['peak_mem_allocated_gb']:.2f}/{r['peak_mem_reserved_gb']:.2f}GB err {r['errors']}  {r['prompt']}")
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
