"""Post training checks for the all-50 paper style reference and task proxies (t59).

1. Velocity cosine along the base's own path. For all 5571 demo frames x 16 draws x 10 Euler steps of
   /workspace/data/t59_distill (x_t, t, v_base on pi0.5's own sampling path), compute v_task and v_ref and report the
   mean cosine of task vs ref, task vs base, ref vs base. Cosine per (frame, draw, step) over the flattened chunk,
   reported over all 32 dims and over the first 7 (LIBERO) dims, plus per step. Paper (App. C.8 / Fig. 10b, measured on
   the steered path): task vs ref 0.865, task vs base 0.767, ref vs base 0.946 (0.935 on the base's own path).
2. Task proxy sampling. From the same fixed noise pi0.5 used for x0.npy (x_t[:, draw, 0], numpy default_rng((59, frame,
   draw))), 10 Euler steps at the stored t values with dt = -0.1, x <- x + dt * v (pps_real.sampler's convention).
   MSE to the demo chunk (t59_task/actions.npy, normalized space, first 7 dims), averaged over frames and draws, next to
   pi0.5's x0 vs the demo chunk. The reference proxy is sampled the same way for context. Also in raw (unnormalized)
   LIBERO action units via the checkpoint's quantile stats.

Usage
  /workspace/openpi/.venv/bin/python scripts/stage2/check_task_paper.py
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from paper_proxy import load_proxy  # noqa: E402


def cos(a, b):
    return F.cosine_similarity(a.flatten(1).double(), b.flatten(1).double(), dim=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", default="/workspace/runs/reference_t59_paper_all50/final.pt")
    p.add_argument("--task", default="/workspace/runs/task_t59_paper_all50/final.pt")
    p.add_argument("--distill", default="/workspace/data/t59_distill")
    p.add_argument("--task-data", default="/workspace/data/t59_task")
    p.add_argument("--images", default="/workspace/data/t59_demos/policy_obs.npz")
    p.add_argument("--out", default="/workspace/runs/task_t59_paper_all50/checks.json")
    p.add_argument("--chunk", type=int, default=16, help="frames per forward chunk")
    p.add_argument("--max-frames", type=int, default=None, help="testing only: first N frames")
    p.add_argument("--mem-fraction", type=float, default=None)
    args = p.parse_args()
    dev = torch.device("cuda")
    if args.mem_fraction:
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0)

    dd, td = pathlib.Path(args.distill), pathlib.Path(args.task_data)
    demo = np.load(dd / "demo_idx.npy")
    z = np.load(args.images)
    assert np.array_equal(z["demo_idx"], demo) and np.array_equal(np.load(td / "demo_idx.npy"), demo)
    images = torch.stack([torch.from_numpy(z["image"]), torch.from_numpy(z["wrist_image"])], 1)
    X = np.load(dd / "x_t.npy", mmap_mode="r")  # (F, D, S, 10, 32)
    V = np.load(dd / "v_base.npy", mmap_mode="r")
    X0 = np.load(dd / "x0.npy", mmap_mode="r")  # (F, D, 10, 32)
    T = torch.from_numpy(np.load(dd / "t.npy")).to(dev)
    A = torch.from_numpy(np.load(td / "actions.npy")).to(dev)  # (F, 10, 32)
    man = json.loads((td / "manifest.json").read_text())
    q01 = torch.tensor(man["actions_q01"][:7], device=dev)
    q99 = torch.tensor(man["actions_q99"][:7], device=dev)

    def unnorm(x):  # openpi Unnormalize._unnormalize_quantile on the first 7 dims
        return (x[..., :7] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01

    Fn, D, S = X.shape[:3]
    if args.max_frames:
        Fn = min(Fn, args.max_frames)
    dt = torch.tensor(-1.0 / S, dtype=torch.float32, device=dev)
    ref, task = load_proxy(args.ref, dev), load_proxy(args.task, dev)

    keys = ["task_ref", "task_base", "ref_base"]
    acc = {f"{k}_{d}": torch.zeros(S, dtype=torch.float64, device=dev) for k in keys for d in ("all32", "first7")}
    n_tuples = 0
    mse = {k: 0.0 for k in ("task_vs_demo", "ref_vs_demo", "pi05_vs_demo", "task_vs_pi05",
                            "task_vs_demo_raw", "ref_vs_demo_raw", "pi05_vs_demo_raw")}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, Fn, args.chunk):
            fr = np.arange(i, min(i + args.chunk, Fn))
            n = len(fr)
            img = images[fr].to(dev)
            kv_r, kv_t = ref.image_kv(ref.encode(img)), task.image_kv(task.encode(img))
            # 1. velocities on the base's own path
            x = torch.from_numpy(np.ascontiguousarray(X[fr])).to(dev).view(-1, 10, 32)  # (n*D*S) frame, draw, step
            vb = torch.from_numpy(np.ascontiguousarray(V[fr])).to(dev).view(-1, 10, 32)
            t = T.repeat(n * D)
            vr = ref.velocity(kv_r, x, t, rep=D * S).float()
            vt = task.velocity(kv_t, x, t, rep=D * S).float()
            step = torch.arange(S, device=dev).repeat(n * D)
            for k, (a, b) in dict(task_ref=(vt, vr), task_base=(vt, vb), ref_base=(vr, vb)).items():
                acc[f"{k}_all32"].index_add_(0, step, cos(a, b))
                acc[f"{k}_first7"].index_add_(0, step, cos(a[..., :7], b[..., :7]))
            n_tuples += n * D
            # 2. sampling from pi0.5's own noise
            xs = {"task": x.view(n, D, S, 10, 32)[:, :, 0].reshape(-1, 10, 32).clone()}
            xs["ref"] = xs["task"].clone()
            for s in range(S):
                ts = T[s].expand(n * D)
                xs["task"] = xs["task"] + dt * task.velocity(kv_t, xs["task"], ts, rep=D).float()
                xs["ref"] = xs["ref"] + dt * ref.velocity(kv_r, xs["ref"], ts, rep=D).float()
            a = A[fr].repeat_interleave(D, 0)[..., :7]
            pi = torch.from_numpy(np.ascontiguousarray(X0[fr])).to(dev).view(-1, 10, 32)
            for name, y in (("task", xs["task"]), ("ref", xs["ref"]), ("pi05", pi)):
                mse[f"{name}_vs_demo"] += float(((y[..., :7] - a) ** 2).mean((1, 2)).sum())
                mse[f"{name}_vs_demo_raw"] += float(((unnorm(y) - unnorm(a)) ** 2).mean((1, 2)).sum())
            mse["task_vs_pi05"] += float(((xs["task"][..., :7] - pi[..., :7]) ** 2).mean((1, 2)).sum())

    res = {"velocity_cosine_base_path": {}, "velocity_cosine_per_step": {}}
    for k, v in acc.items():
        res["velocity_cosine_base_path"][k] = float(v.sum() / (n_tuples * S))
        res["velocity_cosine_per_step"][k] = [float(z) for z in v / n_tuples]
    res["paper_fig10b"] = dict(task_ref=0.865, task_base=0.767, ref_base_steered=0.946, ref_base_own_path=0.935)
    res["sampling_mse_first7"] = {k: v / (Fn * D) for k, v in mse.items()}
    res["sampling_note"] = ("MSE over the 10 x 7 chunk per sample, mean over 5571 frames x 16 draws; same noise for task, "
                            "ref and pi0.5. Unsuffixed: openpi normalized space. _raw: unnormalized LIBERO action units.")
    res["n_frames"], res["draws"], res["steps"] = Fn, D, S
    res["ckpts"] = dict(ref=args.ref, task=args.task)
    pathlib.Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("velocity_cosine_base_path", "sampling_mse_first7")}, indent=2))


if __name__ == "__main__":
    main()
