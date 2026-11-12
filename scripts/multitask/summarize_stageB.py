"""Stage B result for one task (PROTOCOL_multitask.md section a). Plain Python 3 + numpy.

pi0.5 on screening trials 0-49: trials 0-29 from Stage A when the Stage B rerun of trials 0 and 1 reproduced them bit
for bit (<B>/t<T>/repro_compare.json), else the Stage B rerun of 0-49; trials 30-49 from Stage B. task_f on 0-49 from
Stage B. Qualifies if task_f succeeds on 20-42 of 50. Also: paired counts (task_f only / pi0.5 only), exact McNemar p,
and the re-grasp description of the screening trials (an object lifted >= 2 cm, then lost with no finger contact
and back below 1 cm, then grasped again). Writes <B>/t<T>/result.json and prints one line.

Usage: python3 scripts/multitask/summarize_stageB.py --task 50
"""

import argparse
import json
import math
import pathlib

import numpy as np

A = pathlib.Path("/workspace/runs/stage4_screen/stageA")
B = pathlib.Path("/workspace/runs/stage4_screen/stageB")
GPU = "NVIDIA GeForce RTX 4090"
N = 50
LO, HI = 20, 42  # 40% and 85% of 50, both included


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
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(root, task, cell):
    out = {}
    f = root / ("t%d" % task) / cell / "screen.jsonl"
    if f.exists():
        for l in open(f):
            r = json.loads(l)
            out[r["trial"]] = r
    return out


def regrasp(npz):
    """Lifted >= 2 cm, then dropped (no grasp and no finger contact, object back below 1 cm), then a new grasp onset."""
    z = np.load(npz)
    dz, g, a6, fin = z["step_dz"], z["step_grasp"].astype(bool), z["step_a6"], z["step_finger"].astype(bool)
    held = g & (a6 > 0)
    lifted = dropped = False
    for i in range(len(dz)):
        if not lifted:
            lifted = bool(held[i] and dz[i] >= 0.02)
        elif not dropped:
            dropped = bool(not g[i] and not fin[i] and dz[i] < 0.01)
        elif held[i] and (i == 0 or not held[i - 1]):
            return True, True
    return lifted and dropped, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True)
    a = ap.parse_args()
    t = a.task
    rep = json.loads((B / ("t%d" % t) / "repro_compare.json").read_text())
    base_b = load(B, t, "base")
    tf = load(B, t, "task_f")
    if rep["identical_all"]:
        base = dict(load(A, t, "base"))
        base.update({j: r for j, r in base_b.items() if j >= 30})
        base_src = "trials 0-29 Stage A (Stage B rerun of 0 and 1 bit identical), 30-49 Stage B"
    else:
        base = base_b
        base_src = "trials 0-49 Stage B (Stage B rerun of 0 and 1 not identical to Stage A)"
    assert sorted(base) == list(range(N)) and sorted(tf) == list(range(N)), (sorted(base), sorted(tf))
    for d in (base, tf):
        assert all(r["server"]["gpu"] == GPU for r in d.values())
    errors = sum(bool(r["error"]) for d in (base, tf) for r in d.values())
    kb = sum(r["success"] for r in base.values())
    kt = sum(r["success"] for r in tf.values())
    only_t = sum(tf[j]["success"] and not base[j]["success"] for j in range(N))
    only_b = sum(base[j]["success"] and not tf[j]["success"] for j in range(N))
    rg = {}
    for cell, d, root in (("base", base, None), ("task_f", tf, B)):
        drops = regr = 0
        for j, r in d.items():
            src = A if (cell == "base" and rep["identical_all"] and j < 30) else B
            dropped, re = regrasp(src / ("t%d" % t) / cell / "screen" / ("trial_%04d.npz" % j))
            drops += dropped
            regr += re
        rg[cell] = {"trials_with_drop_after_lift": drops, "of_which_regrasped": regr}
    tr = {}
    for kind in ("reference", "task"):
        m = pathlib.Path("/workspace/runs/%s_t%d_faithful_s0/metrics.json" % (kind, t))
        c = pathlib.Path("/workspace/runs/%s_t%d_faithful_s0/config.json" % (kind, t))
        if m.exists():
            mj, cj = json.loads(m.read_text()), json.loads(c.read_text())
            tr[kind] = {"train_min": mj["train_s"] / 60, "gpu": mj["gpu"], "config_gpu": cj["gpu"],
                        "steps": cj["steps"], "batch": cj["batch"], "seed": cj["seed"]}
    res = {"task": t, "task_name": tf[0]["task_name"], "scene": tf[0]["scene"], "n": N,
           "base": {"successes": kb, "rate": kb / N, "wilson95": wilson(kb, N), "source": base_src},
           "task_f": {"successes": kt, "rate": kt / N, "wilson95": wilson(kt, N)},
           "paired": {"task_f_only": only_t, "base_only": only_b, "mcnemar_p": mcnemar(only_t, only_b)},
           "qualifies": LO <= kt <= HI, "rule": "task_f 20-42 of 50", "flag_task_f_100": kt == N,
           "errors": errors, "regrasp_screen": rg, "training": tr, "repro_identical": rep["identical_all"]}
    (B / ("t%d" % t) / "result.json").write_text(json.dumps(res, indent=1))
    print("t%d base %d/50 task_f %d/50 qualifies=%s" % (t, kb, kt, res["qualifies"]))


if __name__ == "__main__":
    main()
