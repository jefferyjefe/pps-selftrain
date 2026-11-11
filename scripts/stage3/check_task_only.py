"""Check before the task proxy alone runs (4090 rerun, step 6): the fast path (pps_steer.task_only_infer, the task
proxy's own Euler loop without the base) must give bit for bit the actions of the reference path (pps_real.sampler
with TaskOnlyHook, the base evaluated but its velocity replaced by v_task). Same 12 t59 demo frames and seeded noise as
check_gamma0.py. Also reports per call time for both paths and how far the task proxy's actions are from pi0.5's.

Exit 1 if any observation differs (the server then runs with --task-impl hook).

Usage
  PYTHONPATH=. /workspace/openpi/.venv/bin/python scripts/stage3/check_task_only.py --out <json>
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pps_real import policy as _policy  # noqa: E402
from pps_steer import PPSSteer, TaskOnlyHook, task_only_infer  # noqa: E402
from serve_eval import REF, TASK  # noqa: E402

PROMPT = "pick up the tomato sauce and put it in the tray"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--images", default="/workspace/data/t59_demos/policy_obs.npz")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    z = np.load(args.images)
    img_all, wr_all, st_all = z["image"], z["wrist_image"], z["state"]
    frames = np.linspace(0, len(img_all) - 1, args.n).round().astype(int)
    sp = _policy.SteerablePolicy(_policy.load_openpi_policy(compile_mode=None))
    steer = PPSSteer(REF, TASK, 0.0)
    hook = TaskOnlyHook(steer)
    rows, ok_all, tm = [], True, {"hook": [], "fast": [], "base": []}
    for f in frames:
        obs = {"observation/image": img_all[f], "observation/wrist_image": wr_all[f],
               "observation/state": st_all[f], "prompt": PROMPT}
        noise = np.random.default_rng((303, int(f))).standard_normal((10, 32)).astype(np.float32)
        torch.cuda.synchronize()
        t0 = time.monotonic()
        hook.begin(obs["observation/image"], obs["observation/wrist_image"])
        a_hook = sp.infer(dict(obs), noise=noise, hook=hook)["actions"]
        torch.cuda.synchronize()
        tm["hook"].append(time.monotonic() - t0)
        t0 = time.monotonic()
        a_fast = task_only_infer(sp, steer, dict(obs), noise)["actions"]
        torch.cuda.synchronize()
        tm["fast"].append(time.monotonic() - t0)
        t0 = time.monotonic()
        a_base = sp.infer(dict(obs), noise=noise)["actions"]
        torch.cuda.synchronize()
        tm["base"].append(time.monotonic() - t0)
        r = {"frame": int(f), "fast_eq_hook": bool(np.array_equal(a_fast, a_hook)),
             "max_abs_fast_minus_hook": float(np.abs(a_fast - a_hook).max()),
             "task_vs_base_max_abs_action_diff": float(np.abs(a_fast - a_base).max()), "shape": list(a_fast.shape)}
        ok_all = ok_all and r["fast_eq_hook"]
        rows.append(r)
        print(json.dumps(r), flush=True)
    res = {"bit_exact_all": ok_all, "n_observations": len(rows), "gpu": torch.cuda.get_device_name(),
           "median_ms": {k: 1000 * float(np.median(v[1:])) for k, v in tm.items()}, "rows": rows}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=2))
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
