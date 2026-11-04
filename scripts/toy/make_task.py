"""Step 2. Define the narrow task variant and save 50 demos for it.

PushT has a single fixed goal pose (256, 256, pi/4), so filtering episodes by goal region
is not possible. The variant instead translates the goal by a fixed shift s along both
axes, an offset of (s, s) px. Demos for the variant are original demos rigidly translated
by the same offset, which is exact because PushT physics is translation invariant away
from the walls. Every translated demo is replayed in the offset goal env to confirm that.

The base should be weak on the variant but not hopeless, so that its prior still carries
useful skill. Candidate shifts are tried from smallest to largest, and the first one where
base success at the 0.90 threshold lands inside the target band is kept.
"""

import argparse

import toy  # noqa F401
import numpy as np

from toy.common import CKPT_DIR, RESULTS_DIR, SEEDS, get_device, load_json, save_json
from toy.data import VARIANT_DEMOS, load_pusht_episodes, save_episodes, translate_episode
from toy.env import fmt, replay, rollout, summarize
from toy.model import load_ckpt
from toy.pps import make_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shifts", type=float, nargs="+", default=[8, 12, 16, 20], help="px along both x and y")
    ap.add_argument("--band", type=float, nargs=2, default=[0.10, 0.30], help="target base success@0.90")
    ap.add_argument("--n-demos", type=int, default=50)
    ap.add_argument("--val-episodes", type=int, default=200)
    ap.add_argument("--min-final-coverage", type=float, default=0.9)
    ap.add_argument("--max-replay-error", type=float, default=1.0, help="pixels")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    dev = get_device(args.device)
    base = load_ckpt(CKPT_DIR / "base.pt", dev)
    full = load_json(RESULTS_DIR / "toy_base_full_task.json")
    print(f"base on full task: {fmt(full)}")

    lo, hi = args.band
    seeds = SEEDS["variant_val"] + np.arange(args.val_episodes)
    tried, chosen = [], None
    for s in sorted(args.shifts):
        offset = (float(s), float(s))
        var = summarize(rollout(make_policy("base", base=base), seeds, goal_offset=offset, device=dev, noise_seed=0))
        ok = lo <= var["success_090"] <= hi
        tried.append({"shift": s, "offset": offset, **var, "in_band": bool(ok)})
        print(f"shift {s:g} px, offset {offset}: base {fmt(var)}, in band {ok}", flush=True)
        if ok and chosen is None:
            chosen = (offset, var)
    save_json({"base_full_task": full, "band_success_090": [lo, hi], "shifts_tried": tried,
               "val_seeds_start": int(seeds[0])}, RESULTS_DIR / "toy_shift_sweep.json")
    if chosen is None:
        raise SystemExit(f"no shift puts base success@0.90 inside [{lo}, {hi}]")
    offset, var = chosen

    episodes = load_pusht_episodes()
    checks = []
    for ep in episodes:
        tep = translate_episode(ep, offset)
        r = replay(tep, goal_offset=offset)
        checks.append((tep, r))
    consistent = [(e, r) for e, r in checks if r["max_kp_error"] < args.max_replay_error]
    eligible = [(e, r) for e, r in consistent if r["final_coverage"] >= args.min_final_coverage]
    print(f"translated demos replaying within {args.max_replay_error}px: {len(consistent)}/{len(episodes)}, "
          f"of which final coverage >= {args.min_final_coverage}: {len(eligible)}")
    if len(eligible) < args.n_demos:
        raise SystemExit(f"only {len(eligible)} eligible demos, need {args.n_demos}")

    rng = np.random.default_rng(args.seed)
    pick = sorted(rng.choice(len(eligible), size=args.n_demos, replace=False))
    demos = [eligible[i][0] for i in pick]
    demo_cov = np.array([eligible[i][1]["final_coverage"] for i in pick])
    save_episodes(demos, VARIANT_DEMOS, goal_offset=np.array(offset))
    print(f"saved {len(demos)} variant demos ({sum(len(d['agent']) for d in demos)} frames) to {VARIANT_DEMOS}")

    save_json({
        "goal_offset": offset,
        "base_full_task": full,
        "base_variant_val": var,
        "n_translated_consistent": len(consistent),
        "n_eligible": len(eligible),
        "demo_source_episodes": [d["episode"] for d in demos],
        "demo_final_coverage_mean": float(demo_cov.mean()),
        "demo_success_090": float(np.mean(demo_cov > 0.90)),
        "demo_success_095": float(np.mean(demo_cov > 0.95)),
        "val_seeds_start": int(seeds[0]),
    }, RESULTS_DIR / "toy_variant.json")
    print(f"variant goal offset {offset}. base success@0.90 full {full['success_090']:.3f} "
          f"vs variant {var['success_090']:.3f}")


if __name__ == "__main__":
    main()
