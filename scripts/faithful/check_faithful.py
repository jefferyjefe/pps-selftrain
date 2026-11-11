"""Step 2 checks for the faithful proxies before any rollout (exit 1 if any fails). openpi venv, RTX 4090.

On N t59 demo observations (policy frames, state, prompt) with seeded noise:
  A. gamma = 0 is pi0.5 bit for bit: stock openpi Policy.infer (eager) == SteerablePolicy.infer with no hook (the
     server's base) == faithful PPS hook at gamma 0 == blend hook at gamma 0, on the 7-dim actions, and the raw
     (10, 32) sampler output of the hooks equals the no-hook output (torch.equal).
  B. Padding dims: at every Euler step of every call, for PPS and blend at gamma 0, 0.2, 0.4, 0.6, 0.8, 1.0 and for
     the task proxy alone (their --only_steer), the velocity used on dims 7..31 equals pi0.5's velocity at that step
     bit for bit (FaithfulSteer counts every step). Informational: whether the raw output's padding dims equal the
     unsteered pi0.5 output's (they cannot for gamma > 0: the base's padding velocity depends on the steered real dims).
  C. Task proxy alone: the fast path (no base) gives the hook path's 7-dim actions bit for bit.
  D. Blend at gamma = 1 equals the task proxy alone bit for bit (so the task-alone cell is the blend's gamma = 1 end).
Usage
  PYTHONPATH=. /workspace/openpi/.venv/bin/python scripts/faithful/check_faithful.py --out <json>
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from faithful_steer import FaithfulSteer, task_only_fast  # noqa: E402
from serve_faithful import REF_F, TASK_F  # noqa: E402

from pps_real import policy as _policy  # noqa: E402
from pps_real import sampler as _sampler  # noqa: E402

PROMPT = "pick up the tomato sauce and put it in the tray"
GAMMAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--ref", default=REF_F)
    p.add_argument("--task", default=TASK_F)
    p.add_argument("--images", default="/workspace/data/t59_demos/policy_obs.npz")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    z = np.load(args.images)
    nf = len(z["image"])
    frames = np.linspace(0, nf - 1, args.n).round().astype(int)
    stock = _policy.load_openpi_policy(compile_mode=None)
    sp = _policy.SteerablePolicy(stock)
    # upstream openpi's PI0Pytorch.__init__ sets the global float32 matmul precision to "high" itself
    # (pi0_pytorch.py:111); pi0.5 runs under that. Loading the proxies must not change it.
    prec = torch.get_float32_matmul_precision()
    assert prec == "high", prec
    steer = FaithfulSteer(args.ref, args.task)
    assert torch.get_float32_matmul_precision() == prec
    rows = []
    ok = {"A": True, "B": True, "C": True, "D": True}
    for f in frames:
        obs = {"observation/image": z["image"][f], "observation/wrist_image": z["wrist_image"][f],
               "observation/state": z["state"][f], "prompt": PROMPT}
        noise = np.random.default_rng((303, int(f))).standard_normal((10, 32)).astype(np.float32)
        inputs, ob = sp.observation(dict(obs))
        st = inputs["state"][0]
        nz = torch.from_numpy(noise)[None].to(sp.device)
        a_stock = stock.infer(dict(obs), noise=noise)["actions"]
        a_base = sp.infer(dict(obs), noise=noise)["actions"]
        raw_base = _sampler.sample_actions(sp.model, sp.device, ob, noise=nz, num_steps=sp.num_steps)
        r = {"frame": int(f), "stock_eq_base": bool(np.array_equal(a_stock, a_base))}
        acts, pad_info = {}, {}
        for mode in ("pps", "blend"):
            for g in GAMMAS:
                v0 = steer.pad_violations
                steer.begin(obs["observation/image"], obs["observation/wrist_image"], st, mode, g)
                a = sp.infer(dict(obs), noise=noise, hook=steer)["actions"]
                st_ = steer.stats()
                steer.begin(obs["observation/image"], obs["observation/wrist_image"], st, mode, g)
                raw = _sampler.sample_actions(sp.model, sp.device, ob, noise=nz, num_steps=sp.num_steps, hook=steer)
                acts[(mode, g)] = a
                pad_info[f"{mode}_{g}"] = {"pad_velocity_violations": steer.pad_violations - v0,
                                           "raw_pad_eq_unsteered": bool(torch.equal(raw[..., 7:], raw_base[..., 7:])),
                                           "raw_eq_unsteered": bool(torch.equal(raw, raw_base)),
                                           "max_abs_action_change": float(np.abs(a - a_base).max()),
                                           "cos_ref_base_real": float(np.nanmean(st_["ref_base_real"]))}
                if g == 0.0:
                    r[f"{mode}0_eq_base"] = bool(np.array_equal(a, a_base))
                    r[f"{mode}0_eq_stock"] = bool(np.array_equal(a, a_stock))
                    r[f"{mode}0_raw_eq_base"] = bool(torch.equal(raw, raw_base))
        v0 = steer.pad_violations
        steer.begin(obs["observation/image"], obs["observation/wrist_image"], st, "only", 1.0)
        a_only = sp.infer(dict(obs), noise=noise, hook=steer)["actions"]
        pad_info["only"] = {"pad_velocity_violations": steer.pad_violations - v0}
        a_fast = task_only_fast(sp, steer, dict(obs), noise)["actions"]
        r["task_fast_eq_hook"] = bool(np.array_equal(a_fast, a_only))
        r["blend1_eq_task_only"] = bool(np.array_equal(acts[("blend", 1.0)], a_only))
        r["pad"] = pad_info
        okA = (r["stock_eq_base"] and r["pps0_eq_base"] and r["pps0_eq_stock"] and r["pps0_raw_eq_base"]
               and r["blend0_eq_base"] and r["blend0_eq_stock"] and r["blend0_raw_eq_base"])
        okB = all(v["pad_velocity_violations"] == 0 for v in pad_info.values())
        ok["A"] &= okA
        ok["B"] &= okB
        ok["C"] &= r["task_fast_eq_hook"]
        ok["D"] &= r["blend1_eq_task_only"]
        assert torch.get_float32_matmul_precision() == prec
        rows.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "pad"}), flush=True)
    res = {"pass": all(ok.values()), "checks": ok, "n_observations": len(rows), "frames": [int(f) for f in frames],
           "pad_steps_checked_total": steer.pad_checked, "pad_violations_total": steer.pad_violations,
           "pi05_matmul_precision": prec, "ref": args.ref, "task": args.task, "gpu": torch.cuda.get_device_name(), "rows": rows}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1), flush=True)
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
