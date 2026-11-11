"""Select the drop test's saved states from the pi0.5 paired trials. Plain Python, any venv.

Rule (deterministic, independent of the order trials finished in): take the shortest prefix of trials 0, 1, 2, ...
that is completely finished and contains --n trials whose mid-carry event fired (and whose snapshot exists) with at
least --min-budget env steps of the trial's budget left after the event (410 - (event step + 1) >= 130 by default),
and use those --n trials in trial order. State id s = position in that list.

The 130 comes from pi0.5's own normal trials only (no drop outcome was used): among the 36 successful pi0.5 trials with
a mid-carry event in trials 0 to 152 (all that had finished when the rule was set, 2026-11-10 09:20 UTC), finishing
from the event took median 46, 90th percentile 77 and at most 120 env steps. Minimum budget = 10 drop frames + 120,
so every pool state leaves room for the slowest undisturbed completion pi0.5 showed after the drop frames.

Exit 0 and write <root>/pool/manifest.json when ready, 2 if not ready yet, 1 if all trials are done and there are
fewer than --n events.
"""

import argparse
import json
import pathlib
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/workspace/runs/eval_t59")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--n-trials", type=int, default=200)
    p.add_argument("--allow-fewer", action="store_true")
    p.add_argument("--min-budget", type=int, default=130,
                   help="env steps left after the event; 410 is the loop end (400 policy steps + 10 settle steps)")
    args = p.parse_args()
    root = pathlib.Path(args.root)
    pool = root / "pool"
    recs = {}
    jl = root / "base" / "paired.jsonl"
    if jl.exists():
        for line in open(jl):
            r = json.loads(line)
            recs[r["trial"]] = r
    chosen, prefix = [], 0
    while prefix < args.n_trials and prefix in recs and len(chosen) < args.n:
        r = recs[prefix]
        if r["event"] and 410 - (r["event"]["step"] + 1) >= args.min_budget and (pool / ("trial_%04d.pkl" % prefix)).exists():
            chosen.append(r)
        prefix += 1
    if len(chosen) < args.n and not args.allow_fewer:
        if prefix >= args.n_trials:
            print("only %d events in all %d trials" % (len(chosen), args.n_trials))
            sys.exit(1)
        print("not ready: %d events in finished prefix 0..%d" % (len(chosen), prefix - 1))
        sys.exit(2)
    man = {"rule": __doc__.split("\n\n")[1], "min_budget_derivation": __doc__.split("\n\n")[2],
           "min_budget": args.min_budget, "prefix_trials": prefix,
           "n_events_in_prefix": len(chosen),
           "states": [{"state": i, "trial": r["trial"], "file": "trial_%04d.pkl" % r["trial"],
                       "init_state": r["init_state"], "event_step": r["event"]["step"],
                       "plan_len": r["event"].get("plan_len"), "event_probe": r["event"]["probe"],
                       "source_trial_success": r["success"]} for i, r in enumerate(chosen)]}
    (pool / "manifest.json").write_text(json.dumps(man, indent=2))
    print("pool ready: %d states from trials %s (prefix 0..%d)" % (len(chosen), [r["trial"] for r in chosen], prefix - 1))


if __name__ == "__main__":
    main()
