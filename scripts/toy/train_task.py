"""Step 4. Initialise the task proxy from the reference proxy and fine tune it on the 50
variant demos with the standard flow matching loss, Eq. (6).

Afterwards, report velocity alignment along the base sampler's path on the variant
observations (the paper's Appendix C.8 check). cos(ref, base) near 1 means condition C1
holds; cos(task, ref) shows how far task supervision moved the proxy.
"""

import argparse

import toy  # noqa F401
import torch
import torch.nn.functional as F

from toy.common import CKPT_DIR, RESULTS_DIR, get_device, save_json, set_seed
from toy.data import VARIANT_DEMOS, load_episodes, make_windows
from toy.model import fm_loss, load_ckpt, save_ckpt
from toy.pps import base_denoising_trajectory
from toy.train import train


@torch.no_grad()
def velocity_alignment(base, ref, task, obs, n: int = 2048, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, len(obs), (n,), generator=g).to(obs.device)
    o = obs[idx]
    noise = torch.randn(n, base.config["act_dim"], generator=g).to(obs.device)
    xs, ks, vb = base_denoising_trajectory(base, o, noise)
    K = xs.shape[0]
    x, k, oo, vb = xs.flatten(0, 1), ks.flatten(), o.repeat(K, 1), vb.flatten(0, 1)
    vr, vt = ref(x, k, oo), task(x, k, oo)
    cos = lambda a, b: float(F.cosine_similarity(a, b, dim=-1).mean())
    return {"cos_ref_base": cos(vr, vb), "cos_task_ref": cos(vt, vr), "cos_task_base": cos(vt, vb),
            "mse_ref_base": float(F.mse_loss(vr, vb)), "residual_norm": float((vt - vr).norm(dim=-1).mean()),
            "base_norm": float(vb.norm(dim=-1).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    dev = get_device(args.device)
    ref_path = CKPT_DIR / f"seed{args.seed}" / "ref.pt"
    task = load_ckpt(ref_path, dev)  # initialised from the reference proxy (its EMA weights)
    for p in task.parameters():
        p.requires_grad_(True)
    demos, _ = load_episodes(VARIANT_DEMOS)
    obs, act = make_windows(demos)
    obs, act = torch.from_numpy(obs).to(dev), torch.from_numpy(act).to(dev)
    print(f"device {dev}, {len(demos)} demos, {len(obs)} training pairs")

    def batch_loss(step):
        idx = torch.randint(0, len(obs), (args.batch,), device=dev)
        return fm_loss(task, obs[idx], act[idx])

    ema, log = train(task, batch_loss, args.steps, args.lr, desc=f"task seed{args.seed}")
    path = CKPT_DIR / f"seed{args.seed}" / "task.pt"
    save_ckpt(path, task, ema, loss_log=log, args=vars(args), init=str(ref_path))
    print(f"saved {path}")

    base = load_ckpt(CKPT_DIR / "base.pt", dev)
    ref = load_ckpt(ref_path, dev)
    task = load_ckpt(path, dev)
    align = velocity_alignment(base, ref, task, obs)
    save_json({"seed": args.seed, **align}, RESULTS_DIR / f"toy_alignment_seed{args.seed}.json")
    print("velocity alignment on base path " + ", ".join(f"{k} {v:.3f}" for k, v in align.items()))


if __name__ == "__main__":
    main()
