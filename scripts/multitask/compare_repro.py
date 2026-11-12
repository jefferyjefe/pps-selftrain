"""Bit-for-bit comparison of rerun normal trials against earlier records. Plain Python 3 + numpy.

Compares <new>/<kind>/trial_XXXX.npz with <old>/<kind>/trial_XXXX.npz (kind paired or screen) for every trial in
<new>'s <kind>.jsonl: every step's action, qpos, qvel and object pose, and the outcome (success, env steps). Writes
<out> (json) and prints one line per trial and IDENTICAL_ALL / NOT_IDENTICAL.

Usage
  python3 scripts/multitask/compare_repro.py --new /workspace/runs/stage4_screen/step0_repro/base \
      --old /workspace/runs/eval_t59_4090/base --out /workspace/runs/stage4_screen/step0_repro/compare.json
"""

import argparse
import json
import pathlib

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--new", required=True)
    p.add_argument("--old", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--kind", default="paired", help="record kind: paired (t59 runner) or screen (run_mt.py)")
    a = p.parse_args()
    new_d, old_d = pathlib.Path(a.new), pathlib.Path(a.old)
    new = {json.loads(l)["trial"]: json.loads(l) for l in open(new_d / (a.kind + ".jsonl"))}
    old = {json.loads(l)["trial"]: json.loads(l) for l in open(old_d / (a.kind + ".jsonl"))}
    rows, ok = [], True
    for t in sorted(new):
        x = np.load(new_d / a.kind / ("trial_%04d.npz" % t))
        y = np.load(old_d / a.kind / ("trial_%04d.npz" % t))
        r = {"trial": t, "success_new": new[t]["success"], "success_old": old[t]["success"],
             "env_steps_new": new[t]["env_steps"], "env_steps_old": old[t]["env_steps"],
             "gpu_new": new[t]["server"].get("gpu"), "gpu_old": old[t]["server"].get("gpu")}
        for k in ("action", "qpos", "qvel", "obj"):
            u, v = x["step_" + k], y["step_" + k]
            r[k + "_identical"] = bool(u.shape == v.shape and np.array_equal(u, v))
        r["identical"] = all(r[k + "_identical"] for k in ("action", "qpos", "qvel", "obj")) and \
            r["success_new"] == r["success_old"] and r["env_steps_new"] == r["env_steps_old"]
        ok &= r["identical"]
        rows.append(r)
        print(json.dumps(r))
    pathlib.Path(a.out).write_text(json.dumps({"identical_all": bool(ok), "new": str(new_d), "old": str(old_d),
                                               "rows": rows}, indent=1))
    print("IDENTICAL_ALL" if ok else "NOT_IDENTICAL")


if __name__ == "__main__":
    main()
