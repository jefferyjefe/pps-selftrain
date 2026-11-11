"""Progress of the faithful evaluation queue: done / total per (item, kind), success so far per cell. Plain Python 3."""

import collections
import json
import pathlib
import subprocess

ROOT = pathlib.Path("/workspace/runs/eval_t59_faithful")


def done_keys(p):
    out = {}
    if p.exists():
        for line in open(p):
            r = json.loads(line)
            out[r["key"]] = r
    return out


def main():
    q = [json.loads(l) for l in open(ROOT / "queue.jsonl")]
    recs = {}
    for c in {j["cell"] for j in q}:
        for kind in ("paired", "drop", "hard"):
            recs[(c, kind)] = done_keys(ROOT / c / (kind + ".jsonl"))
    tot, done = collections.Counter(), collections.Counter()
    for j in q:
        k = (j["item"], j["kind"])
        tot[k] += 1
        key = ("state_%02d_branch_%d" % (j["state"], j["branch"])) if j["kind"] == "drop" else ("trial_%04d" % j["trial"])
        if key in recs[(j["cell"], j["kind"])]:
            done[k] += 1
    for k in sorted(tot):
        print("item %d %-6s %5d / %5d" % (k[0], k[1], done[k], tot[k]))
    print()
    for (c, kind), r in sorted(recs.items()):
        if r:
            s = sum(bool(x["success"]) for x in r.values())
            ms = [x["wall_s"] for x in r.values() if x.get("wall_s")]
            print("%-12s %-6s %4d done, success %3d (%.0f%%), mean wall %.0fs" % (c, kind, len(r), s, 100 * s / len(r),
                                                                           sum(ms) / max(1, len(ms))))
    mp = ROOT / "bank" / "manifest.json"
    if mp.exists():
        print("\nbank states:", len(json.loads(mp.read_text())["states"]))
    print(subprocess.run("nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader", shell=True,
                         capture_output=True, text=True).stdout.strip())
    print(subprocess.run("pgrep -fc 'run_faithful.py [w]orker'", shell=True, capture_output=True, text=True).stdout.strip(),
          "workers;", subprocess.run("pgrep -fc 'serve_faithful.py [-]-port'", shell=True, capture_output=True,
                                     text=True).stdout.strip(), "servers")


if __name__ == "__main__":
    main()
