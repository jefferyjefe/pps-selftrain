"""Summarize the deliverable 3 runs into results/stage2/perturb_summary.json and print a table. Any Python 3.

Checks that the event fired at the right moment, using the paired structure. A perturbed episode and its
unperturbed twin (same init state, env seed and noise) must be identical up to the event, so the event must fire at
the same step in both. At the event the gripper must be commanded closed with both fingers on the bottle, squeezing
and stalled, no earlier step of the logged trace may satisfy the same predicate, and the bottle must not yet be
lifted 2 cm. For teleport the bottle
must have moved the set distance horizontally and not vertically. For release the gripper command must be -1 for the
set number of frames and the fingers must open.

Usage
  python3 scripts/stage2/summarize_perturb.py
"""

import json
import math
import pathlib

DIR = pathlib.Path("results/stage2/perturb")
M0 = pathlib.Path("results/m0/exp1_eval_episodes.jsonl")
OUT = pathlib.Path("results/stage2/perturb_summary.json")
LIFT_M = 0.02  # same lift threshold as eval_perturb.py


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def load(path):
    return {r["episode"]: r for r in (json.loads(x) for x in open(path))} if path.exists() else {}


def main():
    runs = {c: load(DIR / f"{c}.jsonl") for c in ("none", "teleport", "release")}
    m0 = {r["episode"]: r for r in (json.loads(x) for x in open(M0)) if r["task_id"] == 59}
    none = runs["none"]
    res = {"conditions": {}}
    for cond, rs in runs.items():
        eps = sorted(rs)
        n = len(eps)
        k = sum(rs[e]["success"] for e in eps)
        fired = [e for e in eps if rs[e]["fired"]]
        timing = []
        for e in fired:
            r, info = rs[e], rs[e]["event_info"]
            ok = {
                "commanded_closed": info["gripper_action"] > 0,
                "both_fingers_on_bottle": all(any(g.startswith(f"gripper0_finger{i}") for g in info["obj_contacts"])
                                              for i in (1, 2)),
                "squeezing": info["target_width_m"] < info["width_m"] - 0.005,
                "stalled": info["width_m"] - info["width_prev_m"] > -0.002,
                "first_such_step": r["first_event_step_in_trace"] == info["step"],
                "bottle_not_yet_lifted": info["obj_height_above_start_m"] < LIFT_M,
            }
            if cond != "none" and e in none:
                ok["same_step_as_unperturbed_twin"] = none[e]["fired"] and none[e]["event_info"]["step"] == info["step"]
            if cond == "teleport":
                ok["moved_set_distance"] = abs(info["teleport_horizontal_m"] - r["teleport_dist"]) < 1e-6
                ok["no_vertical_move"] = abs(info["teleport_vertical_m"]) < 1e-9
                ok["clear_direction_found"] = info["teleport_clear"]
            if cond == "release":
                w = [x for x in r["trace_window"] if info["step"] < x["t"] <= info["step"] + r["release_frames"]]
                ok["overridden_frames"] = info.get("release_actions_overridden") == r["release_frames"]
                ok["command_open_during_release"] = len(w) == r["release_frames"] and all(x["a6"] == -1.0 for x in w)
                width = [x["grip"][0] - x["grip"][1] for x in r["trace_window"]
                         if info["step"] <= x["t"] <= info["step"] + r["release_frames"]]
                ok["fingers_opened"] = len(width) >= 2 and width[-1] > width[0]
            timing.append({"episode": e, "step": info["step"], **ok})
        all_ok = {key: sum(t[key] for t in timing) for key in (timing[0] if timing else {}) if key not in ("episode", "step")}
        c = {
            "episodes": n, "success": k, "success_rate": k / n if n else None, "wilson95": wilson(k, n),
            "fired": len(fired),
            "success_given_fired": sum(rs[e]["success"] for e in fired),
            "success_rate_given_fired": (sum(rs[e]["success"] for e in fired) / len(fired)) if fired else None,
            "wilson95_given_fired": wilson(sum(rs[e]["success"] for e in fired), len(fired)),
            "timing_checks_passed_of_fired": all_ok,
            "event_steps": {e: rs[e]["event_info"]["step"] for e in fired},
            "lift_after_event": sum(1 for e in fired if rs[e]["lift_step"] is not None
                                    and rs[e]["lift_step"] > rs[e]["event_info"]["step"]),
        }
        if cond != "none":
            paired = [e for e in fired if e in none and none[e]["fired"]]
            c["paired_with_none"] = {
                "episodes": len(paired),
                "none_success": sum(none[e]["success"] for e in paired),
                "perturbed_success": sum(rs[e]["success"] for e in paired),
                "lost_by_perturbation": [e for e in paired if none[e]["success"] and not rs[e]["success"]],
                "gained_under_perturbation": [e for e in paired if rs[e]["success"] and not none[e]["success"]],
            }
            c["regrasped"] = sum(1 for e in fired if rs[e]["regrasp_step"] is not None)
        if cond == "teleport":
            c["teleport_tries"] = {e: rs[e]["event_info"]["teleport_tries"] for e in fired}
        c["infer_ms_mean"] = (sum(rs[e]["infer_ms_mean"] or 0 for e in eps) / n) if n else None
        res["conditions"][cond] = c
    if none:
        common = sorted(set(none) & set(m0))
        res["none_vs_m0_compiled"] = {
            "episodes": len(common),
            "m0_success": sum(m0[e]["success"] for e in common),
            "eager_success": sum(none[e]["success"] for e in common),
            "same_outcome": sum(m0[e]["success"] == none[e]["success"] for e in common),
            "same_outcome_and_steps": sum(m0[e]["success"] == none[e]["success"]
                                          and m0[e]["env_steps"] == none[e]["env_steps"] for e in common),
        }
    OUT.write_text(json.dumps(res, indent=1))
    print(f"{'condition':10} {'success':>12} {'95% CI':>14} {'fired':>6} {'success|fired':>14}")
    for cond, c in res["conditions"].items():
        ci = c["wilson95"]
        print(f"{cond:10} {c['success']:>4}/{c['episodes']:<3} {c['success_rate'] or 0:5.1%} "
              f"{ci[0] or 0:5.1%}-{ci[1] or 0:5.1%} {c['fired']:>6} {c['success_given_fired']:>6}/{c['fired']:<3}")
        print("   timing checks", c["timing_checks_passed_of_fired"])
        if "paired_with_none" in c:
            print("   paired", c["paired_with_none"], "regrasped", c["regrasped"])
    if "none_vs_m0_compiled" in res:
        print("none vs m0 compiled", res["none_vs_m0_compiled"])


if __name__ == "__main__":
    main()
