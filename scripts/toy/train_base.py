"""Step 1. Train the base flow matching policy (the broad generalist) on all of PushT,
then evaluate it on the original task."""

import argparse

import toy  # noqa F401  sets the MPS fallback env var before torch loads
import numpy as np
import torch

from toy.common import CKPT_DIR, RESULTS_DIR, SEEDS, get_device, save_json, set_seed
from toy.data import load_pusht_episodes, make_windows
from toy.env import fmt, rollout, summarize
from toy.model import VelocityNet, fm_loss, load_ckpt, n_params, save_ckpt
from toy.pps import make_policy
from toy.train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-episodes", type=int, default=200)
    ap.add_argument("--device", default=None)
    ap.add_argument("--eval-only", action="store_true", help="skip training and re-evaluate the saved base")
    args = ap.parse_args()

    set_seed(args.seed)
    dev = get_device(args.device)
    path = CKPT_DIR / "base.pt"
    if not args.eval_only:
        fit_base(args, dev, path)
    evaluate(args, dev, path)


def fit_base(args, dev, path):
    episodes = load_pusht_episodes()
    obs, act = make_windows(episodes)
    print(f"device {dev}, {len(episodes)} episodes, {len(obs)} training pairs")
    obs, act = torch.from_numpy(obs).to(dev), torch.from_numpy(act).to(dev)

    net = VelocityNet(width=args.width, depth=args.depth).to(dev)
    print(f"base params {n_params(net):,}")

    def batch_loss(step):
        idx = torch.randint(0, len(obs), (args.batch,), device=dev)
        return fm_loss(net, obs[idx], act[idx])

    ema, log = train(net, batch_loss, args.steps, args.lr, desc="base")
    save_ckpt(path, net, ema, loss_log=log, args=vars(args))
    print(f"saved {path}")


def evaluate(args, dev, path):
    base = load_ckpt(path, dev)
    seeds = SEEDS["full_eval"] + np.arange(args.eval_episodes)
    res = rollout(make_policy("base", base=base), seeds, goal_offset=(0.0, 0.0), device=dev, noise_seed=0)
    summary = summarize(res)
    save_json({"condition": "base on original task", "seeds_start": int(seeds[0]), **summary},
              RESULTS_DIR / "toy_base_full_task.json")
    print(f"base on full task: {fmt(summary)}")


if __name__ == "__main__":
    main()
