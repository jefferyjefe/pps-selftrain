"""Constant baselines for reference proxy fidelity on the t59 distillation dataset. CPU only.

Means come from the training demos only (all demos not in manifest split.val_demos) and are scored on the held out
demos. Two baselines.
  (a) one mean velocity per action dim, shape (32,)
  (b) one mean velocity per (step, horizon position, dim), shape (10, 10, 32)
Each is reported as MSE over all 32 dims and over the first 7 (the dims LIBERO uses), plus per step.

Usage
  /workspace/openpi/.venv/bin/python scripts/stage2/reference_baselines.py [--data DIR]
"""

import argparse
import json
import pathlib

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/workspace/data/t59_distill")
    p.add_argument("--out", default=None, help="default DATA/baselines.json")
    args = p.parse_args()
    d = pathlib.Path(args.data)
    man = json.loads((d / "manifest.json").read_text())
    val_demos = man["split"]["val_demos"]
    demo = np.load(d / "demo_idx.npy")
    v = np.load(d / "v_base.npy")  # (F, draws, step, H, D)
    is_val = np.isin(demo, val_demos)
    vtr, vva = v[~is_val], v[is_val]

    mean_dim = vtr.mean(axis=(0, 1, 2, 3), dtype=np.float64)  # (D,)
    mean_shd = vtr.mean(axis=(0, 1), dtype=np.float64)  # (step, H, D)

    def score(pred):
        err = (vva.astype(np.float64) - pred) ** 2  # broadcast over (Fv, draws, step, H, D)
        return {
            "mse_all32": float(err.mean()),
            "mse_first7": float(err[..., :7].mean()),
            "mse_all32_per_step": err.mean(axis=(0, 1, 3, 4)).tolist(),
            "mse_first7_per_step": err[..., :7].mean(axis=(0, 1, 3, 4)).tolist(),
        }

    out = {
        "data": str(d),
        "val_demos": val_demos,
        "n_train_frames": int((~is_val).sum()),
        "n_val_frames": int(is_val.sum()),
        "n_val_tuples": int(vva.shape[0] * vva.shape[1] * vva.shape[2]),
        "t_values": np.load(d / "t.npy").tolist(),
        "means_from": "training demos only",
        "a_mean_per_dim": score(mean_dim),
        "b_mean_per_step_horizon_dim": score(mean_shd),
        "val_target_var_all32": float(vva.var(dtype=np.float64)),
    }
    path = pathlib.Path(args.out) if args.out else d / "baselines.json"
    path.write_text(json.dumps(out, indent=2))
    for k in ("a_mean_per_dim", "b_mean_per_step_horizon_dim"):
        print(f"{k}: all32 {out[k]['mse_all32']:.4f}  first7 {out[k]['mse_first7']:.4f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
