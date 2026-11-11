"""Safety check before any PPS rollout: with gamma = 0 the PPS code path must give pi0.5's actions bit for bit.

On N t59 demo observations (the 224 px frames the proxies were trained on, with their policy state and the task
prompt) and seeded noise, compare
  A. stock openpi Policy.infer, torch.compile disabled (the eager pi0.5 reference),
  B. SteerablePolicy.infer with no hook (the server's --method base),
  C. SteerablePolicy.infer with the PPS hook at gamma = 0 (proxies encoded and evaluated at all 10 steps),
on the final 7 dim actions (np.array_equal) and, for B vs C, on the raw (10, 32) model output (torch.equal).
Also reports how much gamma = 0.4 changes the actions, the gamma = 0 velocity cosines on these demo frames (training
frames, so ref vs base should be near the 0.9998 of runs/task_t59_paper_all50/checks.json), and per call timing.

Exit code 1 if any gamma = 0 comparison is not bit exact.

Usage
  PYTHONPATH=. /workspace/openpi/.venv/bin/python scripts/stage3/check_gamma0.py
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
from pps_real import sampler as _sampler  # noqa: E402
from pps_steer import PPSSteer  # noqa: E402
from serve_eval import REF, TASK  # noqa: E402

PROMPT = "pick up the tomato sauce and put it in the tray"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--images", default="/workspace/data/t59_demos/policy_obs.npz")
    p.add_argument("--out", default="/workspace/runs/eval_t59/checks/gamma0_check.json")
    args = p.parse_args()
    z = np.load(args.images)
    nf = len(z["image"])
    frames = np.linspace(0, nf - 1, args.n).round().astype(int)

    stock = _policy.load_openpi_policy(compile_mode=None)
    sp = _policy.SteerablePolicy(stock)
    s0 = PPSSteer(REF, TASK, 0.0)
    s4 = PPSSteer(REF, TASK, 0.4)

    rows, ok_all = [], True
    tm = {"base": [], "pps": []}
    for i, f in enumerate(frames):
        obs = {"observation/image": z["image"][f], "observation/wrist_image": z["wrist_image"][f],
               "observation/state": z["state"][f], "prompt": PROMPT}
        noise = np.random.default_rng((303, int(f))).standard_normal((10, 32)).astype(np.float32)
        a_stock = stock.infer(dict(obs), noise=noise)["actions"]
        torch.cuda.synchronize()
        t0 = time.monotonic()
        a_base = sp.infer(dict(obs), noise=noise)["actions"]
        torch.cuda.synchronize()
        tm["base"].append(time.monotonic() - t0)
        t0 = time.monotonic()
        s0.begin(obs["observation/image"], obs["observation/wrist_image"])
        a_pps0 = sp.infer(dict(obs), noise=noise, hook=s0)["actions"]
        torch.cuda.synchronize()
        tm["pps"].append(time.monotonic() - t0)
        st0 = s0.stats()
        s4.begin(obs["observation/image"], obs["observation/wrist_image"])
        a_pps4 = sp.infer(dict(obs), noise=noise, hook=s4)["actions"]
        # raw model output, B vs C
        _, ob = sp.observation(dict(obs))
        nz = torch.from_numpy(noise)[None].to(sp.device)
        raw_b = _sampler.sample_actions(sp.model, sp.device, ob, noise=nz, num_steps=sp.num_steps)
        s0.begin(obs["observation/image"], obs["observation/wrist_image"])
        raw_c = _sampler.sample_actions(sp.model, sp.device, ob, noise=nz, num_steps=sp.num_steps, hook=s0)
        r = {"frame": int(f), "stock_eq_base": bool(np.array_equal(a_stock, a_base)),
             "base_eq_pps0": bool(np.array_equal(a_base, a_pps0)), "stock_eq_pps0": bool(np.array_equal(a_stock, a_pps0)),
             "raw_base_eq_pps0": bool(torch.equal(raw_b, raw_c)),
             "pps04_max_abs_action_change": float(np.abs(a_pps4 - a_base).max()),
             "gamma0_cos_ref_base_first7": float(st0["ref_base_first7"].mean()),
             "gamma0_cos_ref_base_all32": float(st0["ref_base_all32"].mean()),
             "gamma0_cos_task_ref_first7": float(st0["task_ref_first7"].mean()),
             "gamma0_cos_task_ref_all32": float(st0["task_ref_all32"].mean())}
        ok = r["stock_eq_base"] and r["base_eq_pps0"] and r["stock_eq_pps0"] and r["raw_base_eq_pps0"]
        ok_all = ok_all and ok
        rows.append(r)
        print(json.dumps(r), flush=True)

    res = {"bit_exact_all": ok_all, "n_observations": len(rows), "frames": [int(f) for f in frames],
           "source": args.images, "ref": REF, "task": TASK,
           "comparisons": "A stock openpi Policy.infer eager, B SteerablePolicy no hook, C PPS hook gamma 0 "
                          "(proxies evaluated); actions np.array_equal, raw (10,32) torch.equal",
           "pps04_changes_actions_all": all(r["pps04_max_abs_action_change"] > 0 for r in rows),
           "median_ms": {k: 1000 * float(np.median(v[1:])) for k, v in tm.items()},
           "rows": rows}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=2))
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
