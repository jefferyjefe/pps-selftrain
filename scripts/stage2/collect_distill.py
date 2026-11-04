"""Deliverable 2. Collect the reference proxy distillation dataset for t59. Runs in the openpi venv.

Recipe, paper Eq. 5 with Appendix A.3, as in scripts/toy/distill_ref.py. For each demo observation the frozen
base sampler (pi05_libero, our hooked sampler, eager, verified bit exact against stock) is rolled from noise for
its 10 Euler steps, and at every step the tuple (x_t, t, v_base) on the base's own path is recorded. The
observation features are the base's own frozen prefix hidden states, pooled (see docs/stage2.md).

Observation source is every frame of the 50 LIBERO-90 t59 demos, rendered at eval resolution by
scripts/stage2/render_demo_obs.py. Each frame gets --draws independent noise draws, seeded by
numpy default_rng((seed, frame, draw)).

Outputs in --out, all .npy so they can be memory mapped.
  features.npy  (F, 33, 2048) float16  pooled final layer prefix hidden states. Rows 0..15 agentview 4x4 grid,
                rows 16..31 wrist 4x4 grid, row 32 mean over valid language tokens.
  x_t.npy       (F, D, 10, 10, 32) float32  noisy action chunk at each step, x_t[..., 0, :, :] is the noise
  v_base.npy    (F, D, 10, 10, 32) float32  base velocity at each step
  t.npy         (10,) float32  the time value at each step, 1.0 down to 0.1
  x0.npy        (F, D, 10, 32) float32  final sampler output (raw model space, before unnormalize)
  demo_idx.npy, frame_idx.npy  (F,) int
  chunks/        per 256 frame results while collecting, the resume points, removed after the merge
  manifest.json  written at the end

Usage
  /workspace/openpi/.venv/bin/python scripts/stage2/collect_distill.py
"""

import argparse
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import torch

import pps_real
from pps_real import policy as _policy
from pps_real import sampler as _sampler

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from check_sampler import batch_observation  # noqa: E402

N_STEPS = 10
H, A = 10, 32
GRID = 4  # pool each 16x16 image token grid to 4x4
N_IMG_TOK = 256
CHUNK = 256  # frames per chunk file
VAL_DEMOS = list(range(45, 50))  # held out for proxy fidelity checks, a split marker only


def pool_features(prefix_out: torch.Tensor, prefix_pad_masks: torch.Tensor, n_lang: int) -> torch.Tensor:
    """(B, L, 2048) bf16 -> (B, 33, 2048) float32. Prefix layout is 3 image slots of 256 tokens then n_lang tokens.

    Slot 0 is agentview, slot 1 the wrist camera, slot 2 an empty right wrist slot that pi05_libero masks out."""
    po = prefix_out.float()
    b, _, c = po.shape
    s = 16 // GRID
    out = []
    for slot in range(2):
        g = po[:, slot * N_IMG_TOK:(slot + 1) * N_IMG_TOK].reshape(b, 16, 16, c)
        g = g.reshape(b, GRID, s, GRID, s, c).mean(dim=(2, 4)).reshape(b, GRID * GRID, c)
        out.append(g)
    lang = po[:, 3 * N_IMG_TOK:]
    m = prefix_pad_masks[:, 3 * N_IMG_TOK:].float()[..., None]
    assert lang.shape[1] == n_lang
    out.append(((lang * m).sum(1) / m.sum(1))[:, None])
    return torch.cat(out, dim=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--obs", default="/workspace/data/t59_demos/policy_obs.npz")
    p.add_argument("--out", default="/workspace/data/t59_distill")
    p.add_argument("--draws", type=int, default=16)
    p.add_argument("--frames-per-batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=59)
    p.add_argument("--chunk", type=int, default=CHUNK, help="frames per chunk file")
    p.add_argument("--limit", type=int, default=None, help="only the first N frames, for testing")
    args = p.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    z = np.load(args.obs)
    image, wrist, state = z["image"], z["wrist_image"], z["state"]
    demo_idx, frame_idx, prompt = z["demo_idx"], z["frame_idx"], str(z["prompt"])
    F = len(image) if args.limit is None else args.limit
    D = args.draws

    np.save(out / "demo_idx.npy", demo_idx[:F])
    np.save(out / "frame_idx.npy", frame_idx[:F])
    # /workspace is a FUSE network mount where memmap page writeback crawls, so results go to sequentially
    # written chunk files (also the resume points) and are merged into the final .npy files at the end.
    chunk_dir = out / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    fpb = args.frames_per_batch
    assert args.chunk % fpb == 0
    chunks = list(range(0, F, args.chunk))
    todo = [c0 for c0 in chunks if not (chunk_dir / f"chunk_{c0:05d}.npz").exists()]
    print(f"frames {F}, draws {D}, {len(chunks) - len(todo)} of {len(chunks)} chunks done", flush=True)

    pol = _policy.load_openpi_policy(compile_mode=None)
    sp = _policy.SteerablePolicy(pol)
    model, dev = sp.model, sp.device
    t_start = time.time()
    for c0 in todo:
        c_wall0 = time.time()
        c_gpu = 0.0
        acc = {"features": [], "x_t": [], "v_base": [], "x0": []}
        t_ref = None
        for f0 in range(c0, min(F, c0 + args.chunk), fpb):
            fr = list(range(f0, min(F, f0 + fpb)))
            obs = []
            for f in fr:
                o = {"observation/image": image[f], "observation/wrist_image": wrist[f],
                     "observation/state": state[f].astype(np.float64), "prompt": prompt}
                obs += [o] * D
            ob = batch_observation(sp, obs)
            noise = np.stack([np.random.default_rng((args.seed, f, d)).standard_normal((H, A)).astype(np.float32)
                              for f in fr for d in range(D)])
            torch.cuda.synchronize()
            g0 = time.perf_counter()
            rec = _sampler.Recorder()
            x0 = _sampler.sample_actions(model, dev, ob, noise=torch.from_numpy(noise).to(dev), hook=rec)
            xs, ts, vs = rec.stacked()  # (K, B, H, A), (K, B), (K, B, H, A)
            ctx = rec.ctx
            rows0 = torch.arange(len(fr), device=dev) * D  # draw 0 of each frame
            n_lang = ctx.prefix_out.shape[1] - 3 * N_IMG_TOK
            fe = pool_features(ctx.prefix_out[rows0], ctx.prefix_pad_masks[rows0], n_lang)
            torch.cuda.synchronize()
            c_gpu += time.perf_counter() - g0
            t_now = ts[:, 0].cpu().numpy()
            assert t_ref is None or np.array_equal(t_now, t_ref)
            t_ref = t_now
            B = len(fr)
            acc["features"].append(fe.cpu().numpy().astype(np.float16))
            acc["x_t"].append(xs.permute(1, 0, 2, 3).reshape(B, D, N_STEPS, H, A).cpu().numpy())
            acc["v_base"].append(vs.permute(1, 0, 2, 3).reshape(B, D, N_STEPS, H, A).cpu().numpy())
            acc["x0"].append(x0.reshape(B, D, H, A).cpu().numpy())
        tmp = chunk_dir / f"chunk_{c0:05d}.tmp.npz"
        np.savez(tmp, **{k: np.concatenate(v) for k, v in acc.items()}, t=t_ref,
                 gpu_s=np.float64(c_gpu), wall_s=np.float64(time.time() - c_wall0))
        tmp.rename(chunk_dir / f"chunk_{c0:05d}.npz")
        el = time.time() - t_start
        n_done = todo.index(c0) + 1
        print(f"chunk {c0} done, {n_done}/{len(todo)} this run, {el:.0f}s, "
              f"eta {el / n_done * (len(todo) - n_done):.0f}s", flush=True)

    # Merge.
    parts = {k: [] for k in ("features", "x_t", "v_base", "x0")}
    gpu_s = wall_s = 0.0
    t_ref = None
    for c0 in chunks:
        c = np.load(chunk_dir / f"chunk_{c0:05d}.npz")
        for k in parts:
            parts[k].append(c[k])
        assert t_ref is None or np.array_equal(c["t"], t_ref)
        t_ref = c["t"]
        gpu_s += float(c["gpu_s"])
        wall_s += float(c["wall_s"])
    for k, v in parts.items():
        arr = np.concatenate(v)
        assert len(arr) == F
        np.save(out / f"{k}.npy", arr)
    np.save(out / "t.npy", t_ref)
    feats_shape = [F, 2 * GRID * GRID + 1, 2048]
    for c0 in chunks:
        (chunk_dir / f"chunk_{c0:05d}.npz").unlink()
    chunk_dir.rmdir()

    n_traj = F * D
    files = {p_.name: p_.stat().st_size for p_ in sorted(out.glob("*.npy"))}
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    manifest = {
        "task": f"{pps_real.T59_SUITE} task {pps_real.T59_TASK_ID}, {pps_real.T59_NAME}",
        "prompt": prompt,
        "observation_source": f"{F} frames of the 50 LIBERO-90 t59 demos ({pps_real.T59_DEMO_HDF5}), "
                              "sim states re-rendered at 256 px and processed as the eval client does",
        "base": {"checkpoint": _policy.DEFAULT_CHECKPOINT, "config": "pi05_libero", "openpi_commit": "215abfb",
                 "sampler": "pps_real.sampler.sample_actions, eager, Recorder hook, bit exact vs stock eager",
                 "num_steps": N_STEPS, "batch_rows": fpb * D},
        "repo_commit_at_collection": commit,
        "recipe": "paper Eq. 5 and Appendix A.3, on policy. (x_t, t, v_base) at each of the 10 Euler steps along "
                  "the base's own sampling path from fresh Gaussian noise",
        "draws_per_frame": D,
        "noise": f"numpy default_rng(({args.seed}, frame, draw)).standard_normal((10, 32)) float32",
        "n_frames": F, "n_trajectories": n_traj, "n_tuples": n_traj * N_STEPS,
        "t_values": [float(v) for v in t_ref],
        "time_convention": "openpi, t = 1 is noise, t = 0 is data, Euler dt = -0.1, x_(k+1) = x_k + dt * v_k",
        "features": {"file": "features.npy", "dtype": "float16", "shape": feats_shape,
                     "source": "final layer PaliGemma prefix hidden states (after final norm), "
                               "computed once per frame by the base during sampling",
                     "rows": "0..15 agentview tokens average pooled 16x16 to 4x4 (row major), "
                             "16..31 wrist tokens pooled the same way, 32 mean over valid language tokens",
                     "from_row": "draw 0 of each frame"},
        "arrays": {"x_t.npy": "(F, draws, step, 10, 32) float32", "v_base.npy": "(F, draws, step, 10, 32) float32",
                   "x0.npy": "(F, draws, 10, 32) float32 raw model output", "t.npy": "(step,) float32",
                   "demo_idx.npy": "(F,) demo number 0..49", "frame_idx.npy": "(F,) frame within demo"},
        "action_dims": "all 32 model dims are stored. LIBERO uses the first 7, the other 25 are padding the base "
                       "still denoises, so they are part of x_t",
        "split": {"val_demos": VAL_DEMOS, "note": "a marker only, nothing was dropped"},
        "base_forward_passes": {"prefix_passes": n_traj, "denoise_step_passes": n_traj * N_STEPS,
                                "sampler_calls_at_batch": int(np.ceil(F / fpb)),
                                "note": "the prefix is recomputed for every row, so the 16 draws of a frame "
                                        "each pay one prefix pass"},
        "gpu_s_sampling": gpu_s, "wall_s_collection": wall_s,
        "bytes": files, "total_gb": sum(files.values()) / 1e9,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: manifest[k] for k in ("n_trajectories", "n_tuples", "gpu_s_sampling", "total_gb")}, indent=1))


if __name__ == "__main__":
    main()
