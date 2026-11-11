"""Build (or grow) the common drop bank for the faithful evaluation. Plain Python 3, no simulator needed.

Selection, per source, is scripts/stage3/select_pool.py's rule: the shortest prefix of trials 0, 1, 2, ... that is
completely finished and contains 40 trials whose first mid-carry event fired, whose snapshot exists, and that leave
>= 130 env steps of budget after the event (410 - (event step + 1) >= 130); those 40 in trial order.
Sources, in state-id order: base (pi0.5; Nov 11 RTX 4090 normal trials, /workspace/runs/eval_t59_4090/base and
pool_own4090), pps_f_g04 and task_f (this run's normal trials, <root>/<cell>/paired.jsonl, <root>/pool_src/<cell>).
States 0-39 base, 40-79 pps_f_g04, 80-119 task_f. Snapshots are copied into <root>/bank/<source>_trial_XXXX.pkl and
SHA-256 checked. The manifest only ever grows (a source is appended once it and every earlier source are ready), so
state ids never change. Exit 0 when all three sources are in, 2 otherwise.
"""

import hashlib
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path("/workspace/runs/eval_t59_faithful")
N, MIN_BUDGET, N_TRIALS = 40, 130, 200
SOURCES = [("base", pathlib.Path("/workspace/runs/eval_t59_4090/base/paired.jsonl"),
            pathlib.Path("/workspace/runs/eval_t59_4090/pool_own4090")),
           ("pps_f_g04", ROOT / "pps_f_g04/paired.jsonl", ROOT / "pool_src/pps_f_g04"),
           ("task_f", ROOT / "task_f/paired.jsonl", ROOT / "pool_src/task_f")]


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def select(jl, pool):
    recs = {}
    if jl.exists():
        for line in open(jl):
            r = json.loads(line)
            recs[r["trial"]] = r
    chosen, prefix = [], 0
    while prefix < N_TRIALS and prefix in recs and len(chosen) < N:
        r = recs[prefix]
        if r["event"] and 410 - (r["event"]["step"] + 1) >= MIN_BUDGET and (pool / ("trial_%04d.pkl" % prefix)).exists():
            chosen.append(r)
        prefix += 1
    return chosen, prefix


def main():
    bank = ROOT / "bank"
    bank.mkdir(parents=True, exist_ok=True)
    mp = bank / "manifest.json"
    man = json.loads(mp.read_text()) if mp.exists() else {
        "rule": __doc__.split("\n\n")[1], "min_budget": MIN_BUDGET, "per_source": N, "sources": [], "states": []}
    have = [s["source"] for s in man["sources"]]
    for name, jl, pool in SOURCES:
        if name in have:
            continue
        chosen, prefix = select(jl, pool)
        if len(chosen) < N:
            print("%s not ready: %d qualifying in finished prefix 0..%d" % (name, len(chosen), prefix - 1))
            break
        s0 = len(man["states"])
        for i, r in enumerate(chosen):
            src = pool / ("trial_%04d.pkl" % r["trial"])
            dst = bank / ("%s_trial_%04d.pkl" % (name, r["trial"]))
            shutil.copyfile(src, dst)
            h = sha(src)
            assert sha(dst) == h
            man["states"].append({"state": s0 + i, "source": name, "trial": r["trial"], "file": dst.name,
                                  "init_state": r["init_state"], "event_step": r["event"]["step"],
                                  "plan_len": r["event"].get("plan_len"), "event_probe": r["event"]["probe"],
                                  "source_trial_success": r["success"], "sha256": h,
                                  "source_gpu": (r.get("server") or {}).get("gpu")})
        man["sources"].append({"source": name, "states": [s0, s0 + N - 1], "prefix_trials": prefix,
                               "trials": [r["trial"] for r in chosen], "records": str(jl), "pool": str(pool)})
        tmp = mp.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(man, indent=1))
        tmp.replace(mp)
        print("added %s: states %d-%d from trials %s" % (name, s0, s0 + N - 1, [r["trial"] for r in chosen]))
    done = len(man["sources"]) == len(SOURCES)
    print("bank: %d states, sources %s" % (len(man["states"]), [s["source"] for s in man["sources"]]))
    sys.exit(0 if done else 2)


if __name__ == "__main__":
    main()
