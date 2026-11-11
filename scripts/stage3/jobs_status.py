"""Job accounting for the stage 3 workers. Plain Python.

Prints one line: paired_undone paired_unclaimed drop_undone drop_unclaimed. Drop counts are -1 until DROP_OK exists.
With --cleanup, removes claim files whose worker process is gone and whose job is not finished, so the job can be
picked up again (a crashed worker leaves its claim behind).
"""

import argparse
import json
import os
import pathlib

ROOT = pathlib.Path("/workspace/runs/eval_t59")
METHODS = ("base", "pps")
N_PAIRED, N_STATES, N_BRANCHES = 200, 40, 5


def done(path):
    out = set()
    if path.exists():
        for line in open(path):
            out.add(json.loads(line)["key"])
    return out


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cleanup", action="store_true")
    p.add_argument("--root", default=None, help="run directory (default the Nov 10 one)")
    p.add_argument("--methods", nargs="*", default=None, help="methods with jobs (default base pps)")
    args = p.parse_args()
    global ROOT, METHODS
    if args.root:
        ROOT = pathlib.Path(args.root)
    if args.methods:
        METHODS = tuple(args.methods)
    claims = ROOT / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    jobs = {}
    for m in METHODS:
        d = done(ROOT / m / "paired.jsonl")
        for t in range(N_PAIRED):
            k = "trial_%04d" % t
            jobs["paired_%s_%s" % (m, k)] = ("paired", k in d)
    if (ROOT / "POOL_EXTRA").exists() and not (ROOT / "DROP_OK").exists():
        d = done(ROOT / "base" / "paired.jsonl")
        for t in range(N_PAIRED, N_PAIRED + 60):
            k = "trial_%04d" % t
            jobs["paired_base_%s" % k] = ("paired", k in d)
    drop_ok = (ROOT / "DROP_OK").exists()
    if drop_ok:
        n_states = len(json.loads((ROOT / "pool" / "manifest.json").read_text())["states"])
        for m in METHODS:
            d = done(ROOT / m / "drop.jsonl")
            for s in range(n_states):
                for b in range(N_BRANCHES):
                    k = "state_%02d_branch_%d" % (s, b)
                    jobs["drop_%s_%s" % (m, k)] = ("drop", k in d)
    if args.cleanup:
        for f in claims.iterdir():
            if f.name in jobs and not jobs[f.name][1]:
                try:
                    pid = int(f.read_text() or "0")
                except ValueError:
                    pid = 0
                if not alive(pid):
                    f.unlink()
    claimed = {f.name for f in claims.iterdir()}
    cnt = {}
    for kind in ("paired", "drop"):
        und = [n for n, (k, dn) in jobs.items() if k == kind and not dn]
        cnt[kind] = (len(und), len([n for n in und if n not in claimed]))
    if not drop_ok:
        cnt["drop"] = (-1, -1)
    print(cnt["paired"][0], cnt["paired"][1], cnt["drop"][0], cnt["drop"][1])


if __name__ == "__main__":
    main()
