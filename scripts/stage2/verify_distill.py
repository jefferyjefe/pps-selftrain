"""Checks on the distillation dataset written by collect_distill.py. Runs in the openpi venv.

1. Path consistency on every trajectory. x_t[step 0] is the seeded noise, x_(k+1) = x_k + dt * v_k exactly in
   float32, and x0 = x_9 + dt * v_9.
2. Batch 1 replay on a sample of (frame, draw) pairs. The base is called again on that single observation and
   noise, as it will be called at eval time, and its recorded v_base is compared with the dataset. The dataset was
   collected at 64 rows per call, and bf16 kernels give slightly different numbers at different batch sizes, so
   this measures the size of that difference rather than asserting equality.
   A free replay can drift onto a different path, because small differences at early steps grow through the flow.
   So a teacher forced replay is also run. At each step the batch 1 base is evaluated on the recorded x_t (the hook
   returns the velocity that lands the next state on the recorded one), which compares the targets themselves.
3. Features from the batch 1 replay against the stored features.

Usage
  /workspace/openpi/.venv/bin/python scripts/stage2/verify_distill.py [--data DIR] [--n 32]
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

from pps_real import policy as _policy
from pps_real import sampler as _sampler

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from collect_distill import N_IMG_TOK, pool_features  # noqa: E402


def cos(a, b):
    a, b = a.reshape(-1, a.shape[-1] * a.shape[-2]), b.reshape(-1, b.shape[-1] * b.shape[-2])
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/workspace/data/t59_distill")
    p.add_argument("--obs", default="/workspace/data/t59_demos/policy_obs.npz")
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--out", default="results/stage2/distill_check.json")
    args = p.parse_args()
    d = pathlib.Path(args.data)
    man = json.loads((d / "manifest.json").read_text())
    seed = int(man["noise"].split("((")[1].split(",")[0])
    xs = np.load(d / "x_t.npy", mmap_mode="r")
    vs = np.load(d / "v_base.npy", mmap_mode="r")
    x0 = np.load(d / "x0.npy", mmap_mode="r")
    feats = np.load(d / "features.npy", mmap_mode="r")
    F, D = xs.shape[:2]
    dt = np.float32(-0.1)

    # 1. Path consistency, streamed over frames.
    path_bad, noise_bad = 0, 0
    for f0 in range(0, F, 256):
        x, v, o = np.asarray(xs[f0:f0 + 256]), np.asarray(vs[f0:f0 + 256]), np.asarray(x0[f0:f0 + 256])
        nxt = x[:, :, :-1] + dt * v[:, :, :-1]
        path_bad += int((~np.all(nxt == x[:, :, 1:], axis=(2, 3, 4))).sum())
        path_bad += int((~np.all(o == x[:, :, -1] + dt * v[:, :, -1], axis=(2, 3))).sum())
    rng = np.random.default_rng(0)
    sample = [(int(f), int(rng.integers(D))) for f in rng.choice(F, size=min(args.n, F), replace=False)]
    for f, dr in sample:
        nz = np.random.default_rng((seed, f, dr)).standard_normal((10, 32)).astype(np.float32)
        noise_bad += int(not np.array_equal(nz, xs[f, dr, 0]))

    # 2 and 3. Batch 1 replay.
    z = np.load(args.obs)
    image, wrist, state, prompt = z["image"], z["wrist_image"], z["state"], str(z["prompt"])
    sp = _policy.SteerablePolicy(_policy.load_openpi_policy(compile_mode=None))
    model, dev = sp.model, sp.device
    v_abs, v_cos, x0_abs, f_abs, f_cos, tf_abs, tf_cos, tf_rel = [], [], [], [], [], [], [], []
    for f, dr in sample:
        o = {"observation/image": image[f], "observation/wrist_image": wrist[f],
             "observation/state": state[f].astype(np.float64), "prompt": prompt}
        _, ob = sp.observation(o)
        rec = _sampler.Recorder()
        out = _sampler.sample_actions(model, dev, ob, noise=torch.from_numpy(np.array(xs[f, dr, 0]))[None].to(dev),
                                      hook=rec)
        _, _, v1 = rec.stacked()
        v1 = v1[:, 0].cpu().numpy()
        v_abs.append(float(np.abs(v1 - vs[f, dr]).max()))
        v_cos.append(float(cos(v1, np.asarray(vs[f, dr])).min()))
        x0_abs.append(float(np.abs(out[0].cpu().numpy() - x0[f, dr]).max()))
        # Teacher forced. The hook sees v_base at x_t equal to the recorded x_t (up to float32 rounding of the
        # previous step) and steers onto the next recorded state.
        rec_x = torch.from_numpy(np.array(xs[f, dr])).to(dev)
        tf_v = []

        def force(step, x, t, v, c, rec_x=rec_x, tf_v=tf_v):
            tf_v.append(v[0].clone())
            if step + 1 < rec_x.shape[0]:
                return ((rec_x[step + 1] - x[0]) / -0.1)[None]
            return None

        _sampler.sample_actions(model, dev, ob, noise=rec_x[0][None], hook=force)
        tf = torch.stack(tf_v).cpu().numpy()
        tf_abs.append(float(np.abs(tf - vs[f, dr]).max()))
        tf_cos.append(float(cos(tf, np.asarray(vs[f, dr])).min()))
        tf_rel.append(float(np.linalg.norm(tf - vs[f, dr]) / np.linalg.norm(vs[f, dr])))
        ctx = rec.ctx
        fe = pool_features(ctx.prefix_out, ctx.prefix_pad_masks, ctx.prefix_out.shape[1] - 3 * N_IMG_TOK)
        fe = fe[0].cpu().numpy().astype(np.float16).astype(np.float32)
        fs = np.asarray(feats[f]).astype(np.float32)
        f_abs.append(float(np.abs(fe - fs).max()))
        f_cos.append(float(((fe * fs).sum(-1) / (np.linalg.norm(fe, axis=-1) * np.linalg.norm(fs, axis=-1))).min()))
    v_mag = float(np.abs(np.asarray(vs[:64])).mean())
    res = {
        "data": str(d), "n_frames": F, "draws": D,
        "path_inconsistent_trajectories": path_bad, "noise_mismatch_of_sampled": noise_bad,
        "batch1_replay": {
            "n": len(sample),
            "v_base_max_abs_diff": {"max": max(v_abs), "median": float(np.median(v_abs))},
            "v_base_min_cosine_per_step": {"min": min(v_cos), "median": float(np.median(v_cos))},
            "x0_max_abs_diff": {"max": max(x0_abs), "median": float(np.median(x0_abs))},
            "v_base_mean_abs_for_scale": v_mag,
            "free_replay_pairs_with_min_cosine_below_0.999": int(sum(c < 0.999 for c in v_cos)),
            "teacher_forced_v_base_max_abs_diff": {"max": max(tf_abs), "median": float(np.median(tf_abs))},
            "teacher_forced_v_base_min_cosine_per_step": {"min": min(tf_cos), "median": float(np.median(tf_cos))},
            "teacher_forced_relative_l2_error": {"max": max(tf_rel), "median": float(np.median(tf_rel))},
            "features_max_abs_diff": {"max": max(f_abs), "median": float(np.median(f_abs))},
            "features_min_cosine_per_row": {"min": min(f_cos), "median": float(np.median(f_cos))},
        },
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
