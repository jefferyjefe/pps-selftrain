"""Deliverable 1 check. The steering capable sampler must reproduce openpi's stock sampler bit for bit.

Two modes, run in separate processes so the stock compiled policy is built exactly as the m0 server built it.

--mode eager     loads pi05_libero with torch.compile disabled, takes 20 t59 demo observations with fixed
                 noise and compares, with torch.equal on the raw model output (B, 10, 32) and
                 np.array_equal on the final 7 dim actions,
                   stock PI0Pytorch.sample_actions, run twice (determinism of the reference itself)
                   ours with no hook
                   ours with a Recorder hook (returns None)
                   ours with an identity hook (returns v_base)
                   ours with a PPS hook whose delta is exactly zero (v_base + 1.0 * (z - z))
                   openpi Policy.infer against SteerablePolicy.infer (full input and output transforms)
                 at batch 1, plus one batched call of all 20. Also checks that the recorded path satisfies
                 x_(k+1) = x_k + dt * v_k exactly and that a nonzero hook changes the output as specified.
                 Saves raw outputs for the compiled comparison and times both samplers.
--mode compiled  loads pi05_libero with the config's torch.compile (max-autotune), as served in milestone 0,
                 runs the same 20 inputs, reports how far the compiled stock output is from the eager one,
                 and times it.

Usage
  /workspace/openpi/.venv/bin/python scripts/stage2/check_sampler.py --mode eager
  /workspace/openpi/.venv/bin/python scripts/stage2/check_sampler.py --mode compiled
"""

import argparse
import json
import pathlib
import time

import numpy as np
import torch

from pps_real import policy as _policy
from pps_real import sampler as _sampler

OUT_JSON = pathlib.Path("results/stage2/sampler_check.json")
RAW = pathlib.Path("/workspace/runs/stage2/sampler_check_raw.npz")
N_OBS = 20
NOISE_SEED = 2026


def pick_observations(path="/workspace/data/t59_demos/policy_obs.npz"):
    """20 frames, one from each of demos 0, 2, ..., 38, at evenly spread phases of the demo."""
    z = np.load(path)
    picks = []
    for j in range(N_OBS):
        d = 2 * j
        idx = np.where(z["demo_idx"] == d)[0]
        picks.append(int(idx[int((j + 0.5) / N_OBS * len(idx))]))
    image, wrist, state, prompt = z["image"][picks], z["wrist_image"][picks], z["state"][picks], str(z["prompt"])
    obs = [{"observation/image": image[j], "observation/wrist_image": wrist[j],
            "observation/state": state[j].astype(np.float64), "prompt": prompt} for j in range(len(picks))]
    noise = [np.random.default_rng((NOISE_SEED, j)).standard_normal((10, 32)).astype(np.float32) for j in range(N_OBS)]
    return picks, obs, noise


def timed(fn, n=50, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": float(np.median(ts)), "p10_ms": float(np.percentile(ts, 10)),
            "p90_ms": float(np.percentile(ts, 90)), "n": n}


def batch_observation(sp, obs_list):
    obs_b = [sp.observation(o)[1] for o in obs_list]
    fields = {}
    for name in ("images", "image_masks"):
        fields[name] = {k: torch.cat([getattr(o, name)[k] for o in obs_b]) for k in getattr(obs_b[0], name)}
    for name in ("state", "tokenized_prompt", "tokenized_prompt_mask"):
        fields[name] = torch.cat([getattr(o, name) for o in obs_b])
    return type(obs_b[0])(**fields)


def run_eager():
    torch.manual_seed(0)
    pol = _policy.load_openpi_policy(compile_mode=None)
    sp = _policy.SteerablePolicy(pol)
    model, dev = sp.model, sp.device
    assert not hasattr(model.sample_actions, "_torchdynamo_orig_callable"), "stock sampler is compiled"
    picks, obs, noise = pick_observations()

    rows, raw_stock, act_stock = [], [], []
    all_equal = True
    for j, (o, nz) in enumerate(zip(obs, noise)):
        _, observation = sp.observation(o)
        n_t = torch.from_numpy(nz).to(dev)[None]
        stock = model.sample_actions(dev, observation, noise=n_t)
        stock2 = model.sample_actions(dev, observation, noise=n_t)
        ours = _sampler.sample_actions(model, dev, observation, noise=n_t)
        rec = _sampler.Recorder()
        ours_rec = _sampler.sample_actions(model, dev, observation, noise=n_t, hook=rec)
        ours_id = _sampler.sample_actions(model, dev, observation, noise=n_t, hook=lambda s, x, t, v, c: v)
        z = torch.randn(1, 10, 32, device=dev, generator=torch.Generator(dev).manual_seed(j))
        zero_delta = _sampler.pps_hook(lambda x, t, c: z, lambda x, t, c: z, gamma=1.0)
        ours_pps0 = _sampler.sample_actions(model, dev, observation, noise=n_t, hook=zero_delta)

        # Recorded path is the path the sampler took.
        xs, ts, vs = rec.stacked()
        dt = torch.tensor(-1.0 / 10, dtype=torch.float32, device=dev)
        path_ok = all(torch.equal(xs[k + 1], xs[k] + dt * vs[k]) for k in range(9))
        path_ok = path_ok and torch.equal(ours_rec, xs[9] + dt * vs[9]) and torch.equal(xs[0], n_t)

        # Full policy path, input and output transforms included.
        a_stock = pol.infer(dict(o), noise=nz.copy())["actions"]
        a_ours = sp.infer(dict(o), noise=nz.copy())["actions"]
        a_ours_rec = sp.infer(dict(o), noise=nz.copy(), hook=_sampler.Recorder())["actions"]

        r = {
            "obs": j, "frame": picks[j],
            "stock_rerun_equal": torch.equal(stock, stock2),
            "ours_no_hook_equal": torch.equal(stock, ours),
            "ours_recorder_equal": torch.equal(stock, ours_rec),
            "ours_identity_hook_equal": torch.equal(stock, ours_id),
            "ours_pps_zero_delta_equal": torch.equal(stock, ours_pps0),
            "policy_infer_actions_equal": bool(np.array_equal(a_stock, a_ours)),
            "policy_infer_recorder_actions_equal": bool(np.array_equal(a_stock, a_ours_rec)),
            "recorded_path_consistent": bool(path_ok),
            "t_values": [float(t[0]) for t in ts],
        }
        all_equal &= all(v for k, v in r.items() if k.endswith("_equal") or k == "recorded_path_consistent")
        rows.append(r)
        raw_stock.append(stock.cpu().numpy()[0])
        act_stock.append(a_stock)
        print(j, {k: v for k, v in r.items() if isinstance(v, bool)}, flush=True)

    # A nonzero hook changes the output, and the Euler update uses exactly the returned velocity.
    _, observation = sp.observation(obs[0])
    n_t = torch.from_numpy(noise[0]).to(dev)[None]
    c = 0.05
    rec2 = _sampler.Recorder()

    def shifted(step, x, t, v, ctx):
        rec2(step, x, t, v, ctx)
        return v + c

    out_shift = _sampler.sample_actions(model, dev, observation, noise=n_t, hook=shifted)
    xs2, _, vs2 = rec2.stacked()
    dt = torch.tensor(-0.1, dtype=torch.float32, device=dev)
    shift_ok = all(torch.equal(xs2[k + 1], xs2[k] + dt * (vs2[k] + c)) for k in range(9))
    shift_ok = shift_ok and torch.equal(out_shift, xs2[9] + dt * (vs2[9] + c))
    shift_diff = float((out_shift - torch.from_numpy(raw_stock[0]).to(dev)).abs().max())

    # Batched, as the distillation collector runs it.
    obs_b = batch_observation(sp, obs)
    noise_b = torch.from_numpy(np.stack(noise)).to(dev)
    stock_b = model.sample_actions(dev, obs_b, noise=noise_b)
    ours_b = _sampler.sample_actions(model, dev, obs_b, noise=noise_b, hook=_sampler.Recorder())
    batch_equal = torch.equal(stock_b, ours_b)
    b1 = torch.from_numpy(np.stack(raw_stock)).to(dev)
    batch_vs_b1 = float((stock_b - b1).abs().max())

    rec_ctx = _sampler.Recorder()
    _sampler.sample_actions(model, dev, observation, noise=n_t, hook=rec_ctx)
    ctx = rec_ctx.ctx

    _, observation = sp.observation(obs[0])
    timing = {
        "stock_eager_sample_actions": timed(lambda: model.sample_actions(dev, observation, noise=n_t)),
        "ours_eager_no_hook": timed(lambda: _sampler.sample_actions(model, dev, observation, noise=n_t)),
        "ours_eager_recorder_hook": timed(lambda: _sampler.sample_actions(model, dev, observation, noise=n_t,
                                                                          hook=_sampler.Recorder())),
        "ours_policy_infer": timed(lambda: sp.infer(dict(obs[0]), noise=noise[0].copy())),
    }
    RAW.parent.mkdir(parents=True, exist_ok=True)
    np.savez(RAW, raw=np.stack(raw_stock), actions=np.stack(act_stock))
    res = {
        "openpi_commit": "215abfb", "checkpoint": _policy.DEFAULT_CHECKPOINT, "config": "pi05_libero",
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "n_obs": N_OBS, "batch": 1,
        "noise": f"numpy default_rng(({NOISE_SEED}, j)).standard_normal((10, 32))",
        "observations": "one frame from each of t59 demos 0, 2, ..., 38, rendered by scripts/stage2/render_demo_obs.py",
        "all_bit_exact": bool(all_equal and shift_ok and batch_equal),
        "per_obs": rows,
        "nonzero_hook_check": {"hook": f"v_base + {c}", "euler_update_exact": bool(shift_ok),
                               "max_abs_change_in_x0": shift_diff},
        "batched_20": {"stock_vs_ours_equal": bool(batch_equal),
                       "max_abs_diff_batch20_vs_batch1_stock": batch_vs_b1},
        "prefix_out_shape": list(ctx.prefix_out.shape), "prefix_valid_tokens": int(ctx.prefix_pad_masks.sum()),
        "prefix_out_dtype": str(ctx.prefix_out.dtype),
        "timing_eager": timing,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_obs"}, indent=1))


def run_compiled():
    pol = _policy.load_openpi_policy(compile_mode="default")
    model, dev = pol._model, pol._pytorch_device  # noqa: SLF001
    assert hasattr(model.sample_actions, "_torchdynamo_orig_callable"), "stock sampler is not compiled"
    sp = _policy.SteerablePolicy(pol)
    picks, obs, noise = pick_observations()
    ref = np.load(RAW)
    t0 = time.time()
    raw, acts = [], []
    for o, nz in zip(obs, noise):
        _, observation = sp.observation(o)
        raw.append(model.sample_actions(dev, observation, noise=torch.from_numpy(nz).to(dev)[None]).cpu().numpy()[0])
        acts.append(pol.infer(dict(o), noise=nz.copy())["actions"])
    compile_and_first_s = time.time() - t0
    raw, acts = np.stack(raw), np.stack(acts)
    d_raw = np.abs(raw - ref["raw"])
    d_act = np.abs(acts - ref["actions"])
    _, observation = sp.observation(obs[0])
    n_t = torch.from_numpy(noise[0]).to(dev)[None]
    res = json.loads(OUT_JSON.read_text())
    res["compiled_stock_vs_eager_stock"] = {
        "compile_mode": "max-autotune",
        "bit_exact_obs": int(sum(np.array_equal(a, b) for a, b in zip(raw, ref["raw"]))),
        "raw_max_abs_diff": float(d_raw.max()), "raw_mean_abs_diff": float(d_raw.mean()),
        "actions_max_abs_diff": float(d_act.max()), "actions_mean_abs_diff": float(d_act.mean()),
        "actions_max_abs_diff_per_dim": d_act.max(axis=(0, 1)).tolist(),
        "action_abs_mean_per_dim": np.abs(ref["actions"]).mean(axis=(0, 1)).tolist(),
        "wall_s_including_compile": compile_and_first_s,
    }
    res["timing_compiled"] = {
        "stock_compiled_sample_actions": timed(lambda: model.sample_actions(dev, observation, noise=n_t)),
        "stock_compiled_policy_infer": timed(lambda: pol.infer(dict(obs[0]), noise=noise[0].copy())),
    }
    e = res["timing_eager"]["ours_eager_recorder_hook"]["median_ms"]
    c = res["timing_compiled"]["stock_compiled_sample_actions"]["median_ms"]
    res["slowdown_ours_eager_vs_stock_compiled"] = e / c
    OUT_JSON.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("compiled_stock_vs_eager_stock", "timing_compiled",
                                          "slowdown_ours_eager_vs_stock_compiled")}, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["eager", "compiled"], required=True)
    m = p.parse_args().mode
    run_eager() if m == "eager" else run_compiled()
