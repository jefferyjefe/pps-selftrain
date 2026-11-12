"""Harder-starts pilot summary (PROTOCOL_multitask.md addendum 1). Plain Python 3.

Per task: pi0.5 (base) and task_f success on the kept harder starts, Wilson 95%, the paired 2x2 table and the exact
McNemar p, next to the t59 harder-start numbers (results/eval_t59_faithful/REPORT.md). Writes
/workspace/runs/stage4_screen/hard/t<T>/result.json and prints one line per task.

Usage: python3 scripts/multitask/summarize_hard.py --task 14 77
"""

import argparse
import json
import math
import pathlib

H = pathlib.Path("/workspace/runs/stage4_screen/hard")
GPU = "NVIDIA GeForce RTX 4090"
T59 = {"source": "results/eval_t59_faithful/REPORT.md, Harder starts (bottle 7 cm off the demo range)", "n": 50,
       "base": 29, "task_f": 14, "task_f_only": 7, "base_only": 22}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def one(t):
    d = H / ("t%d" % t)
    man = json.loads((d / "manifest.json").read_text())
    if man["skipped"]:
        res = {"task": t, "skipped": True, "skip_reason": man["skip_reason"],
               "tried": [{k: x[k] for k in ("direction", "unit_xy", "shift_m", "n_pass")} for x in man["tried"]]}
        (d / "result.json").write_text(json.dumps(res, indent=1))
        print("t%d SKIPPED: %s" % (t, man["skip_reason"]))
        return
    n = man["n_kept"]
    cells = {}
    for c in ("base", "task_f"):
        recs = {}
        for l in open(d / c / "hard.jsonl"):
            r = json.loads(l)
            recs[r["trial"]] = r
        assert sorted(recs) == list(range(n)), (c, sorted(recs))
        assert all(r["server"]["gpu"] == GPU and r["server"]["task_id"] in (t, None) for r in recs.values())
        cells[c] = recs
    b, f = cells["base"], cells["task_f"]
    kb = sum(r["success"] for r in b.values())
    kf = sum(r["success"] for r in f.values())
    both = sum(b[j]["success"] and f[j]["success"] for j in range(n))
    bo = sum(b[j]["success"] and not f[j]["success"] for j in range(n))
    fo = sum(f[j]["success"] and not b[j]["success"] for j in range(n))
    tried = [{k: x[k] for k in ("direction", "unit_xy", "shift_m", "n_pass")} for x in man["tried"]]
    res = {"task": t, "skipped": False, "direction": man["direction"], "tried": tried,
           "shift_m": tried[-1]["shift_m"], "n": n, "beyond_demo_range_m": man["beyond_demo_range_m"],
           "base": {"successes": kb, "rate": kb / n, "wilson95": wilson(kb, n)},
           "task_f": {"successes": kf, "rate": kf / n, "wilson95": wilson(kf, n)},
           "paired": {"both": both, "base_only": bo, "task_f_only": fo, "neither": n - both - bo - fo,
                      "mcnemar_p": mcnemar(fo, bo)},
           "errors": sum(bool(r["error"]) for c in cells.values() for r in c.values()),
           "t59_harder_starts": T59}
    (d / "result.json").write_text(json.dumps(res, indent=1))
    print("t%d hard (%s, %.2f m, n=%d): base %d/%d task_f %d/%d paired both %d base_only %d task_f_only %d" % (
        t, man["direction"], res["shift_m"], n, kb, n, kf, n, both, bo, fo))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, nargs="+", required=True)
    for t in ap.parse_args().task:
        one(t)


if __name__ == "__main__":
    main()
