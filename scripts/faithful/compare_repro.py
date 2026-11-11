"""Does pi0.5 on this pod reproduce the Nov 11 RTX 4090 normal trials bit for bit? Plain Python 3 + numpy.

Compares <root>/repro_check/base/paired/trial_XXXX.npz (rerun now through serve_faithful.py's base method and
run_faithful.py) with /workspace/runs/eval_t59_4090/base/paired/trial_XXXX.npz: every step's action, qpos, qvel and
bottle pose, and the outcome. All identical -> REUSE_OK (the Nov 11 pi0.5 normal trials and their snapshots are used).
Otherwise REUSE_FAIL, and pi0.5 normal trials 0-199 are inserted into the main queue after the item-1 normal trials.
"""

import json
import pathlib

import numpy as np

ROOT = pathlib.Path("/workspace/runs/eval_t59_faithful")
OLD = pathlib.Path("/workspace/runs/eval_t59_4090/base")


def main():
    new = {json.loads(l)["trial"]: json.loads(l) for l in open(ROOT / "repro_check/base/paired.jsonl")}
    old = {json.loads(l)["trial"]: json.loads(l) for l in open(OLD / "paired.jsonl")}
    rows, ok = [], True
    for t in sorted(new):
        a = np.load(ROOT / "repro_check/base/paired" / ("trial_%04d.npz" % t))
        b = np.load(OLD / "paired" / ("trial_%04d.npz" % t))
        r = {"trial": t, "success_new": new[t]["success"], "success_old": old[t]["success"],
             "env_steps_new": new[t]["env_steps"], "env_steps_old": old[t]["env_steps"],
             "gpu_new": new[t]["server"].get("gpu"), "gpu_old": old[t]["server"].get("gpu")}
        for k in ("action", "qpos", "qvel", "obj"):
            x, y = a["step_" + k], b["step_" + k]
            r[k + "_identical"] = bool(x.shape == y.shape and np.array_equal(x, y))
        r["identical"] = all(r[k + "_identical"] for k in ("action", "qpos", "qvel", "obj")) and \
            r["success_new"] == r["success_old"] and r["env_steps_new"] == r["env_steps_old"]
        ok &= r["identical"]
        rows.append(r)
        print(json.dumps(r))
    (ROOT / "repro_check/compare.json").write_text(json.dumps({"identical_all": ok, "rows": rows}, indent=1))
    if ok:
        (ROOT / "repro_check/REUSE_OK").touch()
        print("REUSE_OK")
        return
    (ROOT / "repro_check/REUSE_FAIL").touch()
    q = [json.loads(l) for l in open(ROOT / "queue.jsonl")]
    last = max(i for i, j in enumerate(q) if j["kind"] == "paired" and j["item"] == 1)
    ins = [{"kind": "paired", "cell": "base", "trial": t, "item": 1} for t in range(200)]
    q = q[:last + 1] + ins + q[last + 1:]
    tmp = ROOT / "queue.jsonl.tmp"
    tmp.write_text("".join(json.dumps(j) + "\n" for j in q))
    tmp.replace(ROOT / "queue.jsonl")
    print("REUSE_FAIL: pi0.5 normal trials 0-199 queued")


if __name__ == "__main__":
    main()
