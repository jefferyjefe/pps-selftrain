"""Step 5 drop calibration helpers (PROTOCOL_multitask.md section c). Plain Python 3 + numpy.

  select --task T   pilot states: the first 30 pilot trials (trial order) of pi0.5 whose first mid-carry event left
                    >= 130 env steps of budget (410 - (event step + 1) >= 130) and whose snapshot exists. Writes
                    <root>/t<T>/pilot_states.json. Prints the number found and the pilot trials done.
  summarize --task T
                    per level L: pi0.5 recovery, completion, re-grasp rate and teleport outcome over the pilot states;
                    the frozen level (lowest L with recovery in [60%, 85%], else closest to 72.5%, tie lower L).
                    Writes <root>/t<T>/cal_result.json.
"""

import argparse
import json
import math
import pathlib

ROOT = pathlib.Path("/workspace/runs/stage4_screen/step5")
N_STATES = 30
MAX_PILOTS = 150
MIN_BUDGET = 130
T_END = 410
LEVELS_M = (0.0, 0.03, 0.06, 0.09, 0.12, 0.15)
BAND = (0.60, 0.85)
MID = 0.725


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def select(task):
    d = ROOT / ("t%d" % task)
    recs = {}
    for l in open(d / "base" / "pilot.jsonl"):
        r = json.loads(l)
        recs[r["trial"]] = r
    n_done = 0
    while n_done in recs:
        n_done += 1  # pilot trials 0..n_done-1 all present
    states = []
    for j in range(n_done):
        r = recs[j]
        ev = r["event"]
        if ev is None or T_END - (ev["step"] + 1) < MIN_BUDGET:
            continue
        f = d / "pilot_src" / "base" / ("trial_%04d.pkl" % j)
        assert f.exists(), f
        states.append({"trial": j, "cell": "base", "file": f.name, "event_step": ev["step"],
                       "pilot_success": r["success"]})
        if len(states) == N_STATES:
            break
    out = {"task": task, "n_states": len(states), "pilot_trials_done": n_done, "states": states,
           "rule": "first %d pilot trials (offset 30000) whose first mid-carry event leaves >= %d steps" % (N_STATES, MIN_BUDGET)}
    (d / "pilot_states.json").write_text(json.dumps(out, indent=1))
    print(len(states), n_done)


def summarize(task):
    d = ROOT / ("t%d" % task)
    man = json.loads((d / "pilot_states.json").read_text())
    n = man["n_states"]
    recs = [json.loads(l) for l in open(d / "base" / "cal.jsonl")]
    assert all(r["server"]["gpu"] == "NVIDIA GeForce RTX 4090" for r in recs)
    levels = []
    for L, dist in enumerate(LEVELS_M):
        rs = {r["state"]: r for r in recs if r["level"] == L}
        assert sorted(rs) == list(range(n)), (L, sorted(rs))
        k_rec = sum(r["recovered"] for r in rs.values())
        k_suc = sum(r["success"] for r in rs.values())
        k_rg = sum(r["regrasp_step"] is not None for r in rs.values())
        moved = sum(bool((r["disturb"] or {}).get("moved")) for r in rs.values())
        levels.append({"level": L, "dist_m": dist, "n": n, "recovery": k_rec / n, "recovery_k": k_rec,
                       "recovery_wilson95": wilson(k_rec, n), "completion": k_suc / n, "completion_k": k_suc,
                       "regrasp": k_rg / n, "regrasp_k": k_rg, "teleport_moved_k": moved,
                       "errors": sum(bool(r["error"]) for r in rs.values())})
    inband = [x for x in levels if BAND[0] <= x["recovery"] <= BAND[1]]
    if inband:
        frozen, why = inband[0], "lowest level with pi0.5 recovery in [60%, 85%]"
    else:
        frozen = min(levels, key=lambda x: (abs(x["recovery"] - MID), x["level"]))
        why = "no level in [60%, 85%]: closest to 72.5% (tie lower level)"
    out = {"task": task, "pilot_states": n, "pilot_trials_done": man["pilot_trials_done"], "levels": levels,
           "frozen_level": frozen["level"], "frozen_dist_m": frozen["dist_m"], "why": why,
           "flag_pi05_never_regrasps": frozen["regrasp_k"] == 0}
    (d / "cal_result.json").write_text(json.dumps(out, indent=1))
    for x in levels:
        print("t%d L%d d=%.2f recovery %d/%d completion %d/%d regrasp %d/%d moved %d" % (
            task, x["level"], x["dist_m"], x["recovery_k"], n, x["completion_k"], n, x["regrasp_k"], n,
            x["teleport_moved_k"]))
    print("t%d FROZEN L%d (%s)" % (task, frozen["level"], why))


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("select", "summarize"))
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--root", default=str(ROOT))
    a = ap.parse_args()
    ROOT = pathlib.Path(a.root)
    (select if a.cmd == "select" else summarize)(a.task)


if __name__ == "__main__":
    main()
