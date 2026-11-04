"""Step 5. Roll out a steered policy on the variant.

The sampler itself lives in toy/pps.py (velocity_fn and make_policy) so that eval.py uses
exactly the same code. Modes
  pps   v_base + gamma * (v_task - v_ref) at every Euler step. gamma 0 is the base.
  base  frozen base only.
  task  task proxy alone.
  ref   reference proxy alone (diagnostic).
"""

import argparse

import toy  # noqa F401
import numpy as np

from toy.common import SEEDS, get_device
from toy.data import variant_offset
from toy.env import fmt, rollout, summarize
from toy.pps import MODES, load_policies, make_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=MODES, default="pps")
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0, help="proxy training seed to load")
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--full-task", action="store_true", help="evaluate on the original goal instead")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    dev = get_device(args.device)
    base, ref, task = load_policies(dev, args.seed)
    offset = (0.0, 0.0) if args.full_task else variant_offset()
    policy = make_policy(args.mode, base=base, task=task, ref=ref, gamma=args.gamma)
    seeds = SEEDS["steer_debug"] + np.arange(args.episodes)
    s = summarize(rollout(policy, seeds, goal_offset=offset, device=dev, noise_seed=args.noise_seed))
    tag = f"pps gamma {args.gamma}" if args.mode == "pps" else args.mode
    print(f"{tag} on goal offset {offset}: {fmt(s)}")


if __name__ == "__main__":
    main()
