"""Does this server's v3 ours path reproduce the Nov 11 RTX 4090 v3 normal trials bit for bit? Plain Python 3 + numpy.

Compares <root>/pps_v3_g04/paired/trial_XXXX.npz and <root>/task_v3/paired/trial_XXXX.npz (rerun now through
serve_faithful.py's pps_v3 / task_v3 methods) with /workspace/runs/eval_t59_4090/{pps,task}/paired/trial_XXXX.npz:
every step's action, qpos, qvel and bottle pose, and the outcome. Writes <root>/checks/v3_repro.json; all identical
-> <root>/checks/V3_REUSE_OK, and summarize_faithful.py then uses the Nov 11 v3 normal trials (same trials, env seeds
and noise seeds) for the faithful vs v3 comparison on normal trials.
"""

import json
import pathlib

import numpy as np

ROOT = pathlib.Path("/workspace/runs/eval_t59_faithful")
OLD = pathlib.Path("/workspace/runs/eval_t59_4090")
PAIRS = (("pps_v3_g04", "pps"), ("task_v3", "task"))


def main():
    rows, ok = [], True
    for new_c, old_c in PAIRS:
        new = {json.loads(l)["trial"]: json.loads(l) for l in open(ROOT / new_c / "paired.jsonl")}
        old = {json.loads(l)["trial"]: json.loads(l) for l in open(OLD / old_c / "paired.jsonl")}
        for t in sorted(new):
            a = np.load(ROOT / new_c / "paired" / ("trial_%04d.npz" % t))
            b = np.load(OLD / old_c / "paired" / ("trial_%04d.npz" % t))
            r = {"cell": new_c, "nov11_cell": old_c, "trial": t, "success_new": new[t]["success"],
                 "success_old": old[t]["success"], "env_steps_new": new[t]["env_steps"],
                 "env_steps_old": old[t]["env_steps"], "gpu_new": new[t]["server"].get("gpu"),
                 "gpu_old": old[t]["server"].get("gpu")}
            for k in ("action", "qpos", "qvel", "obj"):
                x, y = a["step_" + k], b["step_" + k]
                r[k + "_identical"] = bool(x.shape == y.shape and np.array_equal(x, y))
            r["identical"] = all(r[k + "_identical"] for k in ("action", "qpos", "qvel", "obj")) and \
                r["success_new"] == r["success_old"] and r["env_steps_new"] == r["env_steps_old"]
            ok &= r["identical"]
            rows.append(r)
            print(json.dumps(r))
    (ROOT / "checks" / "v3_repro.json").write_text(json.dumps({"identical_all": ok, "rows": rows}, indent=1))
    if ok and rows:
        (ROOT / "checks" / "V3_REUSE_OK").touch()
        print("V3_REUSE_OK")
    else:
        print("V3 NOT REPRODUCED")


if __name__ == "__main__":
    main()
