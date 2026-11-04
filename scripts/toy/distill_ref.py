"""Step 3. Distill the reference proxy from the frozen base on the variant's observations.

Default recipe is the paper's on policy velocity distillation, Eq. (5) and Appendix A.3.
For each demo observation the frozen base sampler is rolled from noise for its 10 Euler
steps, and the proxy regresses the base velocity at every visited (x_k, k). k is not
sampled; supervision sits exactly on the noise levels and states the base sampler visits.

--recipe offpolicy is an alternative for ablation. It noises the demo action at a random
k drawn from the shifted Beta(1.5, 1) and regresses the base velocity there.
"""

import argparse

import toy  # noqa F401
import torch
import torch.nn.functional as F

from toy.common import CKPT_DIR, get_device, set_seed
from toy.data import VARIANT_DEMOS, load_episodes, make_windows
from toy.model import VelocityNet, load_ckpt, n_params, sample_k, save_ckpt
from toy.pps import base_denoising_trajectory
from toy.train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--recipe", choices=["onpolicy", "offpolicy"], default="onpolicy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    dev = get_device(args.device)
    base = load_ckpt(CKPT_DIR / "base.pt", dev)
    for p in base.parameters():
        p.requires_grad_(False)
    demos, _ = load_episodes(VARIANT_DEMOS)
    obs, act = make_windows(demos)
    obs, act = torch.from_numpy(obs).to(dev), torch.from_numpy(act).to(dev)
    print(f"device {dev}, {len(obs)} variant observations, recipe {args.recipe}")

    ref = VelocityNet(width=args.width, depth=args.depth).to(dev)
    print(f"proxy params {n_params(ref):,} (base {n_params(base):,})")

    def batch_loss(step):
        idx = torch.randint(0, len(obs), (args.batch,), device=dev)
        o = obs[idx]
        if args.recipe == "onpolicy":
            xs, ks, vs = base_denoising_trajectory(base, o, torch.randn_like(act[idx]))
            K = xs.shape[0]
            return F.mse_loss(ref(xs.flatten(0, 1), ks.flatten(), o.repeat(K, 1)), vs.flatten(0, 1))
        a = act[idx]
        k = sample_k(len(a), dev)
        x = k[:, None] * torch.randn_like(a) + (1 - k[:, None]) * a
        with torch.no_grad():
            target = base(x, k, o)
        return F.mse_loss(ref(x, k, o), target)

    ema, log = train(ref, batch_loss, args.steps, args.lr, desc=f"ref seed{args.seed}")
    path = CKPT_DIR / f"seed{args.seed}" / "ref.pt"
    save_ckpt(path, ref, ema, loss_log=log, args=vars(args))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
