"""Stage A shortlist, exactly the rule in results/multitask/PROTOCOL_multitask.md section a. Plain Python 3.

Reads <stageA>/t<T>/base/screen.jsonl for every candidate (30 screening trials each, all must be present) and writes
<stageA>/shortlist.json with every task's count, Wilson 95% interval, eligibility and the reason it is or is not on
the shortlist.

Usage: python3 scripts/multitask/select_shortlist.py --stageA /workspace/runs/stage4_screen/stageA
"""

import argparse
import json
import math
import pathlib

CANDIDATES = (2, 14, 19, 31, 38, 43, 50, 56, 60, 64, 70, 77, 82)
N_TRIALS = 30
LO, HI = 3, 15  # 10% and 50% of 30, both included
TARGET = 0.30
AVOID_SCENE = "LIVING_ROOM_SCENE3"
MAX_SHORT = 5


def wilson(k, n, z=1.959964):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stageA", required=True)
    a = ap.parse_args()
    root = pathlib.Path(a.stageA)
    rows = []
    for t in CANDIDATES:
        recs = {}
        for l in open(root / ("t%d" % t) / "base" / "screen.jsonl"):
            r = json.loads(l)
            recs[r["trial"]] = r
        assert sorted(recs) == list(range(N_TRIALS)), (t, sorted(recs))
        assert all(r["server"]["gpu"] == "NVIDIA GeForce RTX 4090" for r in recs.values())
        assert not any(r["error"] for r in recs.values()), t
        k = sum(bool(r["success"]) for r in recs.values())
        rows.append({"task": t, "task_name": recs[0]["task_name"], "scene": recs[0]["scene"], "successes": k,
                     "n": N_TRIALS, "rate": k / N_TRIALS, "wilson95": wilson(k, N_TRIALS),
                     "eligible": LO <= k <= HI, "dist30": abs(k - round(TARGET * N_TRIALS)),  # integer: ties are exact
                     "trials_with_event": sum(r["event"] is not None for r in recs.values())})
    key = lambda r: (r["dist30"], r["task"])  # noqa: E731
    best_in_scene = {}
    for r in sorted([r for r in rows if r["eligible"]], key=key):
        best_in_scene.setdefault(r["scene"], r)
    for r in rows:
        if not r["eligible"]:
            r["why"] = "not eligible: %d/30 outside 3-15" % r["successes"]
        elif best_in_scene[r["scene"]] is not r:
            r["why"] = "eligible, but t%d is closer to 30%% in %s" % (best_in_scene[r["scene"]]["task"], r["scene"])
    pool = sorted(best_in_scene.values(), key=key)
    main_ = [r for r in pool if r["scene"] != AVOID_SCENE][:MAX_SHORT]
    avoid = [r for r in pool if r["scene"] == AVOID_SCENE]
    short = list(main_)
    if len(short) < MAX_SHORT and avoid:
        short += avoid[: MAX_SHORT - len(short)]
    for r in pool:
        if r in short:
            r["why"] = "shortlisted, position %d" % (short.index(r) + 1)
        elif r["scene"] == AVOID_SCENE:
            r["why"] = "eligible, left out: %s preferred against and 5 other scenes qualify" % AVOID_SCENE
        else:
            r["why"] = "eligible, left out: 5 tasks closer to 30%"
    out = {"rule": "eligible 3-15 of 30; one per scene (closest to 30%, tie lower id); up to 5, scenes other than "
                   "LIVING_ROOM_SCENE3 first by |rate-0.30| (tie lower id), LIVING_ROOM_SCENE3 only to fill, placed last",
           "shortlist": [r["task"] for r in short], "tasks": rows}
    (root / "shortlist.json").write_text(json.dumps(out, indent=1))
    for r in rows:
        print("t%-3d %-20s %2d/30 [%.0f, %.0f] events %2d  %s" % (r["task"], r["scene"], r["successes"],
              100 * r["wilson95"][0], 100 * r["wilson95"][1], r["trials_with_event"], r["why"]))
    print("SHORTLIST", out["shortlist"])


if __name__ == "__main__":
    main()
