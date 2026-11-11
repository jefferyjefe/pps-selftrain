"""Two checks on the trained reference proxy. Neither runs pi0.5; everything comes from the stored distillation data.

Time convention, checked in openpi 215abfb src/openpi/models_pytorch/pi0_pytorch.py. Training noises as
x_t = t * noise + (1 - t) * a with target u_t = noise - a (forward, lines 327-329). Sampling starts at x = noise,
t = 1.0, and steps x <- x + dt * v, t <- t + dt with dt = -0.1 in float32 while t >= -dt / 2 (lines 402-419).
So the clean action estimate at (x_t, t, v) is a_hat = x_t - t * v, since x_t - t * (noise - a) = a.

Test 1, does the reference use the camera features. Held out demos, teacher forced on the stored base path.
  a. normal inputs (must reproduce the held out MSE in metrics.json)
  b. camera tokens (rows 0..31) shuffled across held out frames with a fixed seed, language token (row 32), x_t, t kept
  c. camera tokens set to zero, language token kept
  c_all. all 33 tokens set to zero (extra; the language token is a final layer output, so it also carries image info)
Velocity MSE and cosine over all 32 dims and the first 7, overall and per step. The same on the clean action
estimate a_hat, with fraction of variance of a_hat_base explained by a_hat_ref on the first 7 dims.

Test 2, free running sampling. For each frame and each of its 16 draws the starting noise is regenerated from the
manifest recipe and checked bit exact against the stored x_t at step 0. Euler integrating the stored v_base from it
must reproduce the stored path and x0.npy. Then the reference runs all 10 Euler steps on its own output and its final
action is compared with x0 on the first 7 dims, with the gap to the stored path at every step. The yardstick is the
spread of the base's own final actions across the 16 draws of a frame. Run on the held out demos and on 5 train demos.

Usage
  /workspace/openpi/.venv/bin/python scripts/stage2/eval_reference.py
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from train_reference import GPUData, evaluate, load_reference  # noqa: E402

N_CAM_TOK = 32
TRAIN_DEMOS_T2 = [0, 10, 20, 30, 40]


def predict(net, feats_of_frame, data, ids, chunk=16384):
    """v_ref for tuple ids, with feats_of_frame(frame ids) -> (B, 33, 2048) supplying the observation."""
    out = []
    for i in range(0, len(ids), chunk):
        b = ids[i:i + chunk]
        _, x, t, _ = data.batch(b)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(net(feats_of_frame(b // data.per), x, t).float())
    return torch.cat(out)


def cos_rows(a, b):
    return F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1)


def per_step_mean(vals, step, S):
    s = torch.zeros(S, device=vals.device, dtype=torch.float64).index_add_(0, step, vals.double())
    n = torch.zeros(S, device=vals.device, dtype=torch.float64).index_add_(0, step, torch.ones_like(vals, dtype=torch.float64))
    return [float(z) for z in s / n], float(s.sum() / n.sum())


def fve(pred, target, step, S):
    """1 - SSE / SST per step, SST around the per (horizon, dim) mean of target at that step. Pooled overall."""
    per, sse_tot, sst_tot = [], 0.0, 0.0
    for k in range(S):
        m = step == k
        p, y = pred[m].double(), target[m].double()
        sse = ((p - y) ** 2).sum()
        sst = ((y - y.mean(0, keepdim=True)) ** 2).sum()
        per.append(float(1 - sse / sst))
        sse_tot, sst_tot = sse_tot + float(sse), sst_tot + float(sst)
    return per, 1 - sse_tot / sst_tot


def test1_metrics(v_ref, v_base, x, t, step, S):
    r = {}
    for name, (p, y) in {"velocity": (v_ref, v_base),
                         "clean_action": (x - t[:, None, None] * v_ref, x - t[:, None, None] * v_base)}.items():
        e = (p - y) ** 2
        m = {}
        m["mse_all32_per_step"], m["mse_all32"] = per_step_mean(e.mean((1, 2)), step, S)
        m["mse_first7_per_step"], m["mse_first7"] = per_step_mean(e[..., :7].mean((1, 2)), step, S)
        m["cos_all32_per_step"], m["cos_all32"] = per_step_mean(cos_rows(p, y), step, S)
        m["cos_first7_per_step"], m["cos_first7"] = per_step_mean(cos_rows(p[..., :7], y[..., :7]), step, S)
        if name == "clean_action":
            m["fve_first7_per_step"], m["fve_first7"] = fve(p[..., :7], y[..., :7], step, S)
        r[name] = m
    return r


def test1(net, data, metrics_ref, seed):
    ids = data.val_ids
    _, x, t, v_base = data.batch(ids)
    step = ids % data.steps
    val_frames = torch.unique(ids // data.per)
    # shuffle camera tokens across held out frames, fixed seed
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = val_frames[torch.randperm(len(val_frames), generator=g).to(val_frames.device)]
    src = torch.arange(data.feats.shape[0], device=val_frames.device)
    src[val_frames] = perm

    def normal(fr):
        return data.feats[fr]

    def shuffled(fr):
        f = data.feats[fr].clone()
        f[:, :N_CAM_TOK] = data.feats[src[fr], :N_CAM_TOK]
        return f

    def zero_cam(fr):
        f = data.feats[fr].clone()
        f[:, :N_CAM_TOK] = 0
        return f

    def zero_all(fr):
        return torch.zeros_like(data.feats[fr])

    out = {"n_tuples": len(ids), "n_frames": len(val_frames),
           "shuffle": {"seed": seed, "tokens": f"rows 0..{N_CAM_TOK - 1} (camera), row 32 (language) kept",
                       "frames_mapped_to_themselves": int((perm == val_frames).sum()),
                       "frames_mapped_within_same_demo": None}}
    demo = torch.from_numpy(np.load(pathlib.Path(data_dir) / "demo_idx.npy")).to(perm.device)
    out["shuffle"]["frames_mapped_within_same_demo"] = int((demo[perm] == demo[val_frames]).sum())

    # reproduction check against training, both through the training script's evaluate() and this script's path
    ev = evaluate(net, data, ids)
    out["repro"] = {"metrics_json_val_mse_all32": metrics_ref, "train_script_evaluate_mse_all32": ev["mse_all32"]}
    for name, fn in {"a_normal": normal, "b_shuffled_camera": shuffled, "c_zero_camera": zero_cam,
                     "c_all_zero_all_tokens": zero_all}.items():
        out[name] = test1_metrics(predict(net, fn, data, ids), v_base, x, t, step, data.steps)
        if name == "a_normal":
            out["repro"]["this_script_mse_all32"] = out[name]["velocity"]["mse_all32"]
            rel = abs(out[name]["velocity"]["mse_all32"] - metrics_ref) / metrics_ref
            out["repro"]["rel_diff"] = rel
            print(f"repro: metrics.json {metrics_ref:.6f}, evaluate() {ev['mse_all32']:.6f}, "
                  f"here {out[name]['velocity']['mse_all32']:.6f}", flush=True)
            if rel > 0.02:
                raise SystemExit(f"held out MSE does not reproduce (rel diff {rel:.3%}), stopping")
    return out


@torch.no_grad()
def test2(net, data, x0_all, frames, seed_noise, chunk=4096):
    D, S, dev = data.draws, data.steps, data.feats.device
    dt = np.float32(-0.1)
    # 1. noise regeneration, bit exact against stored x_t at step 0
    noise = np.stack([np.random.default_rng((seed_noise, int(f), d)).standard_normal((10, 32)).astype(np.float32)
                      for f in frames for d in range(D)])  # (T, 10, 32), trajectory order (frame, draw)
    traj = (torch.from_numpy(frames).to(dev)[:, None] * D + torch.arange(D, device=dev)[None]).flatten()  # f*D + d
    tup = traj[:, None] * S + torch.arange(S, device=dev)[None]  # (T, S) tuple ids
    xs_base = data.x[tup].cpu().numpy()  # (T, S, 10, 32)
    vs_base = data.v[tup].cpu().numpy()
    x0 = x0_all[frames].reshape(-1, 10, 32)
    noise_match = np.array_equal(noise, xs_base[:, 0])
    if not noise_match:
        raise SystemExit("regenerated noise does not equal stored x_t[:, :, 0], indexing needs working out")
    # 2. Euler integrate stored v_base from the regenerated noise, float32
    x = noise.copy()
    path_maxdiff = 0.0
    for k in range(S):
        if k > 0:
            path_maxdiff = max(path_maxdiff, float(np.abs(x - xs_base[:, k]).max()))
        x = x + dt * vs_base[:, k]
    euler = {"noise_bit_exact": bool(noise_match),
             "path_max_abs_diff": path_maxdiff, "path_bit_exact": path_maxdiff == 0.0,
             "x0_max_abs_diff": float(np.abs(x - x0).max()), "x0_bit_exact": bool(np.array_equal(x, x0))}
    # 3. reference runs alone from the same noise
    tvals = data.t
    ref_path = []  # states at k = 0..10
    for i in range(0, len(traj), chunk):
        tr = traj[i:i + chunk]
        feats = data.feats[tr // D]
        xr = torch.from_numpy(noise[i:i + chunk]).to(dev)
        states = [xr]
        for k in range(S):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = net(feats, xr, tvals[k].expand(len(tr))).float()
            xr = xr + torch.tensor(dt, device=dev) * v
            states.append(xr)
        ref_path.append(torch.stack(states, 1).cpu().numpy())
    ref_path = np.concatenate(ref_path).astype(np.float64)  # (T, 11, 10, 32)
    base_path = np.concatenate([xs_base, x0[:, None]], 1).astype(np.float64)
    gap = (ref_path - base_path) ** 2
    a_ref, a_base = ref_path[:, -1, :, :7], base_path[:, -1, :, :7]
    err = a_ref - a_base

    def cos(a, b):
        a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
        return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))

    # yardstick: the base's own spread across the 16 draws of a frame
    xb = a_base.reshape(len(frames), D, 10, 7)
    var_dim = xb.var(1, ddof=1).mean((0, 1))  # per dim, mean over frames and horizon
    iu = np.triu_indices(D, 1)
    pair_d = xb[:, iu[0]] - xb[:, iu[1]]  # (Fr, pairs, 10, 7)
    pair_cos = cos(xb[:, iu[0]].reshape(-1, 10, 7), xb[:, iu[1]].reshape(-1, 10, 7))
    ref_mse_dim = (err ** 2).mean((0, 1))
    pair_mse_dim = (pair_d ** 2).mean((0, 1, 2))
    return {
        "n_frames": len(frames), "n_trajectories": len(traj), "euler_check": euler,
        "final_first7": {
            "ref_vs_base_mse": float((err ** 2).mean()),
            "ref_vs_base_cos": float(cos(a_ref, a_base).mean()),
            "ref_vs_base_cos_median": float(np.median(cos(a_ref, a_base))),
            "ref_vs_base_mse_per_dim": ref_mse_dim.tolist(),
            "ref_vs_base_mae_per_dim": np.abs(err).mean((0, 1)).tolist(),
            "ref_vs_base_mse_per_trajectory_p50_p90_p99_max":
                np.percentile((err ** 2).mean((1, 2)), [50, 90, 99, 100]).tolist(),
            "base_draw_pair_mse": float(pair_mse_dim.mean()),
            "base_draw_pair_mse_per_dim": pair_mse_dim.tolist(),
            "base_draw_pair_cos": float(pair_cos.mean()),
            "base_draw_var_per_dim": var_dim.tolist(),
            "base_draw_var": float(var_dim.mean()),
            "ratio_ref_mse_to_draw_pair_mse": float((err ** 2).mean() / pair_mse_dim.mean()),
            "ratio_per_dim": (ref_mse_dim / pair_mse_dim).tolist(),
            "base_final_action_var_across_all_trajectories": float(a_base.var(0).mean()),
        },
        "path_gap_per_state": {
            "note": "state k = 0..10, k = 0 is the shared noise, k = 10 is the final action",
            "mse_first7": gap[..., :7].mean((0, 2, 3)).tolist(),
            "mse_all32": gap.mean((0, 2, 3)).tolist(),
        },
    }


def main():
    global data_dir
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/workspace/data/t59_distill")
    p.add_argument("--ckpt", default="/workspace/runs/reference_t59/best.pt")
    p.add_argument("--metrics", default="/workspace/runs/reference_t59/metrics.json")
    p.add_argument("--out", default="/workspace/runs/reference_t59/eval_checks.json")
    p.add_argument("--shuffle-seed", type=int, default=0)
    args = p.parse_args()
    data_dir = args.data
    d = pathlib.Path(args.data)
    man = json.loads((d / "manifest.json").read_text())
    seed_noise = int(man["noise"].split("((")[1].split(",")[0])
    dev = torch.device("cuda")
    data = GPUData(d, dev)
    net = load_reference(args.ckpt, dev)
    metrics_ref = json.loads(pathlib.Path(args.metrics).read_text())["val"]["mse_all32"]

    res = {"ckpt": args.ckpt, "data": args.data, "t_values": data.t.tolist(),
           "convention": "openpi 215abfb pi0_pytorch.py: x_t = t*noise + (1-t)*a, u = noise - a, so a_hat = x_t - t*v; "
                         "sampling x <- x + dt*v from t = 1.0, dt = -0.1 float32"}
    with torch.no_grad():
        res["test1_heldout"] = test1(net, data, metrics_ref, args.shuffle_seed)
    print("test1 done", flush=True)
    demo = np.load(d / "demo_idx.npy")
    x0_all = np.load(d / "x0.npy").reshape(len(demo), data.draws, 10, 32)
    res["test2"] = {"noise_recipe": man["noise"] + ", frame is the global frame index 0..F-1"}
    for name, demos in {"heldout": data.val_demos, "train_demos_" + "_".join(map(str, TRAIN_DEMOS_T2)): TRAIN_DEMOS_T2}.items():
        frames = np.nonzero(np.isin(demo, demos))[0]
        res["test2"][name] = dict(demos=demos, **test2(net, data, x0_all, frames, seed_noise))
        r = res["test2"][name]
        print(f"test2 {name}: euler {r['euler_check']}, final mse7 {r['final_first7']['ref_vs_base_mse']:.5f} "
              f"vs draw pair {r['final_first7']['base_draw_pair_mse']:.5f}", flush=True)
    pathlib.Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
