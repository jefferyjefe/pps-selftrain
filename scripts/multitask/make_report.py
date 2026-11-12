"""Write results/multitask/SCREEN_REPORT.md from results/multitask/screen_summary.json (collect_summary.py) and the
pod's logs. Plain Python 3. Run from the repo root after collect_summary.py.
"""

import datetime as dt
import json
import pathlib
import re

OUT = pathlib.Path("results/multitask")
S = json.loads((OUT / "screen_summary.json").read_text())


def pct(ci):
    return "[%.0f, %.0f]" % (100 * ci[0], 100 * ci[1])


def kn(k, n, ci):
    return "%d/%d = %.0f%% %s" % (k, n, 100.0 * k / n, pct(ci))


def ts(l):
    return dt.datetime.strptime(l.split()[0], "%Y-%m-%dT%H:%M:%SZ")


def walls():
    """(task, seed, kind) -> list of (start, end or None, end line) from chain.log and rest.log."""
    lines = S["logs"]["chain"] + S["logs"]["rest"]
    out = {}
    for i, l in enumerate(lines):
        m = re.search(r" t(\d+) s(\d) (reference|task): start", l)
        if not m:
            continue
        key = (int(m.group(1)), int(m.group(2)), m.group(3))
        end = None
        for l2 in lines[i + 1:]:
            if re.search(r" t%s s%s %s(:| FAILED)" % key, l2) and "start" not in l2:
                end = l2
                break
        out.setdefault(key, []).append((ts(l), ts(end) if end else None, end))
    return out


def main():
    A = S["stageA"]
    B = S["stageB"]
    H = S["hard_pilot"]
    W = walls()
    L = []
    w = L.append
    w("# Stage 4 task screen: report")
    w("")
    w("Branch `stage4-task-screen`, made from `stage3-eval-t59` (89bd01c). The protocol is "
      "`results/multitask/PROTOCOL_multitask.md` (b00a2c1) with addendum 1 (0abab63, the harder-starts pilot). "
      "Every rollout and every training run below ran on one RunPod pod with an %s; the GPU name is in every "
      "training config and every result record. Screening results are for task selection only, not the main "
      "comparison." % S["gpu"])
    w("")
    w("Stage B was cut short by the Nov 12 change of plan (pod stopped by 17:00 UTC; no job started unless "
      "estimated to end by 16:40 UTC). This report makes no task selection.")
    w("")
    # ------------------------------------------------------------------ step 0
    s0 = S.get("step0_repro_t59") or {}
    w("## Step 0")
    w("")
    w("π0.5 on t59 normal trials 0 and 10 reproduced the Nov 11 RTX 4090 records bit for bit (actions, qpos, qvel, "
      "object pose, outcome): `checks/step0_repro_t59.json`%s." % (
          " (IDENTICAL_ALL = %s)" % s0.get("IDENTICAL_ALL", s0.get("identical_all")) if s0 else ""))
    w("")
    # ------------------------------------------------------------------ stage A
    w("## Stage A: π0.5 alone, 30 screening trials per candidate")
    w("")
    w("Seeds: screening offset 20,000 (trial j: init state j mod 50, env.seed(7 + 20,000 + j), noise "
      "default_rng((7, 20,000 + j))). 2 servers × 3 workers. Wilson 95% intervals.")
    w("")
    w("| Task | Scene | Prompt | π0.5 success | Trials with a mid-carry event | Decision |")
    w("| --- | --- | --- | --- | --- | --- |")
    for r in sorted(A["tasks"], key=lambda r: r["task"]):
        name = r["task_name"].split("_", 3)[-1] if "SCENE" in r["task_name"] else r["task_name"]
        name = r["task_name"][len(r["scene"]) + 1:].replace("_", " ")
        w("| t%d | %s | %s | %s | %d | %s |" % (r["task"], r["scene"], name, kn(r["successes"], r["n"], r["wilson95"]),
                                              r["trials_with_event"], r["why"]))
    w("")
    w("Rule (protocol section a): %s." % A["rule"])
    w("")
    w("Shortlist, in order: %s. Only t60 and t64 share a scene among the eligible tasks (LIVING_ROOM_SCENE4); "
      "t60 (14/30) is closer to 30%% than t64 (15/30). Of the 7 eligible tasks, the 5 closest to 30%% are kept; "
      "t60 is the 6th. No LIVING_ROOM_SCENE3 task was eligible (t56: 28/30)." % ", ".join(
          "t%d" % t for t in A["shortlist"]))
    w("")
    # ------------------------------------------------------------------ stage B
    w("## Stage B: faithful proxies seed 0, 50 screening trials")
    w("")
    w("Per candidate in shortlist order: faithful reference (20k steps, batch 8) and task proxy (24k steps, batch "
      "64), seed 0, the t59 recipe on the task's own 50 demos; `check_faithful.py` (checks A–D); π0.5 trials 0 "
      "and 1 rerun and compared with Stage A; then `task_f` on trials 0–49 and π0.5 on 30–49 (0–29 reused from "
      "Stage A when the rerun is bit identical). Qualifies if `task_f` succeeds on 20–42 of 50 (40–85%).")
    w("")
    w("| Task | π0.5 | task_f | task_f only / π0.5 only (McNemar p) | Rerun of π0.5 0–1 vs Stage A | "
      "Drop after lift (re-grasped): π0.5, task_f | Qualifies |")
    w("| --- | --- | --- | --- | --- | --- | --- |")
    for t in sorted(B, key=lambda x: A["shortlist"].index(int(x))):
        r = B[t]
        rg = r["regrasp_screen"]
        w("| t%s | %s | %s | %d / %d (%.2g) | %s | %d (%d), %d (%d) | %s%s |" % (
            t, kn(r["base"]["successes"], 50, r["base"]["wilson95"]),
            kn(r["task_f"]["successes"], 50, r["task_f"]["wilson95"]), r["paired"]["task_f_only"],
            r["paired"]["base_only"], r["paired"]["mcnemar_p"], "identical" if r["repro_identical"] else "NOT identical",
            rg["base"]["trials_with_drop_after_lift"], rg["base"]["of_which_regrasped"],
            rg["task_f"]["trials_with_drop_after_lift"], rg["task_f"]["of_which_regrasped"],
            "yes" if r["qualifies"] else "no", " (task_f 100%: FLAG)" if r["flag_task_f_100"] else ""))
    w("")
    if S["stageB_not_evaluated"]:
        w("Not evaluated: %s." % ", ".join("t%s (%s)" % kv for kv in S["stageB_not_evaluated"].items()))
        w("")
    nq = len(S["stageB_qualifying_so_far"])
    w("Qualifying so far: %s. Not run in Stage B: %s." % (
        ", ".join("t%d" % t for t in S["stageB_qualifying_so_far"]) or "none",
        ", ".join("t%d" % t for t in A["shortlist"] if str(t) not in B and t not in B) or "none"))
    w("")
    # ------------------------------------------------------------------ hard pilot
    w("## Harder-starts pilot (addendum 1): π0.5 and task_f only")
    w("")
    w("A pilot to inform a possible protocol amendment; it does not select tasks. PPS and the blend were not run. "
      "Construction (fixed in addendum 1 before any rollout): the carried object moved along the world table axis "
      "closest to \"away from the goal\", by the demo range width on that axis + 2 cm, rounded up to a whole cm; "
      "the t59 settle check; at least 40 of 50 starts must pass, else the perpendicular axes (+90°, then −90°). "
      "Seeds: offset 40,000. The t59 row is from `results/eval_t59_faithful/REPORT.md` (task proxy alone = the t59 "
      "faithful task proxy).")
    w("")
    w("| Task | Object, axis, shift | Starts passing | Beyond demo range | π0.5 | task_f | Both / π0.5 only / "
      "task_f only / neither | McNemar p |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for t in sorted(H, key=int):
        h = H[t]
        g = h["geometry"]
        axn = lambda u: ("+" if sum(u) > 0 else "−") + ("x" if abs(u[0]) > abs(u[1]) else "y")
        tried = "; ".join("%s (%s) %.0f cm: %d/50" % (x["direction"], axn(x["unit_xy"]), 100 * x["shift_m"],
                                                      x["n_pass"]) for x in h["tried"])
        r = h["result"]
        if h["skipped"] or r is None:
            w("| t%s | %s | %s | — | %s | | | |" % (t, g["obj"], tried, "skipped" if h["skipped"] else "not run"))
            continue
        p = r["paired"]
        w("| t%s | %s, %s, %.0f cm | %s | %.1f–%.1f cm | %s | %s | %d / %d / %d / %d | %.2g |" % (
            t, g["obj"], axn(h["tried"][-1]["unit_xy"]), 100 * r["shift_m"], tried, 100 * r["beyond_demo_range_m"][0],
            100 * r["beyond_demo_range_m"][1], kn(r["base"]["successes"], r["n"], r["base"]["wilson95"]),
            kn(r["task_f"]["successes"], r["n"], r["task_f"]["wilson95"]), p["both"], p["base_only"],
            p["task_f_only"], p["neither"], p["mcnemar_p"]))
    w("| t59 (Nov 11) | tomato_sauce_1, −y, 7 cm | 50/50 | 2.1–7.0 cm | 29/50 = 58% [44, 71] | "
      "14/50 = 28% [18, 42] | — / 22 / 7 / — | 0.0081 |")
    w("")
    for t in sorted(H, key=int):
        r = H[t]["result"]
        if r and not r["skipped"]:
            ks = (B.get(t) or B.get(int(t)) or {})
            if ks:
                w("- t%s, normal screening trials for comparison: π0.5 %d/50, task_f %d/50." % (
                    t, ks["base"]["successes"], ks["task_f"]["successes"]))
    w("")
    # ------------------------------------------------------------------ training
    w("## Training runs (all on the %s)" % S["gpu"])
    w("")
    w("Train time is the training loop (metrics.json `train_s`); wall time is start to end in the chain log "
      "(includes loading the teacher, the frames and, for the reference, building the prefix cache).")
    w("")
    w("| Run | Steps, batch, torch seed | Train time | Wall time | Final probe | Status |")
    w("| --- | --- | --- | --- | --- | --- |")
    notes = {(82, 0, "reference", 0): "stopped by me at 07:15 UTC during setup, before any training step "
                                      "(it would have hit the memory limit; see below)"}
    for key, runs in sorted(S["training"].items()):
        for kind, r in runs.items():
            m = re.match(r"t(\d+)_s(\d)", key)
            k3 = (int(m.group(1)), int(m.group(2)), kind) if m else None
            wl = W.get(k3, []) if (m and "Nov 11" not in key) else []
            for i, (a, b, e) in enumerate(wl):
                wall = "%s–%s UTC%s" % (a.strftime("%H:%M"), b.strftime("%H:%M") if b else "…",
                                        (" (%d min)" % round((b - a).total_seconds() / 60)) if b else "")
                last = i == len(wl) - 1
                if e and "FAILED" in e:
                    st = "killed at the pod's 125 GB memory limit while allocating the prefix cache, before any " \
                         "training step (see below)" if kind == "reference" else e.split(" ", 1)[-1]
                    w("| %s_%s, attempt %d | | | %s | | %s |" % (kind, key, i + 1, wall, st))
                elif k3 + (i,) in notes:
                    w("| %s_%s, attempt %d | | | %s | | %s |" % (kind, key, i + 1, wall.replace("…", "07:15"),
                                                               notes[k3 + (i,)]))
                elif last and "train_min" in r:
                    pr = r["final_probe"] or {}
                    probe = ("mse %.4f" % pr.get("mse_real", pr.get("mse"))) if pr else "—"
                    w("| %s_%s | %d, %d, %d | %.1f min | %s | %s | %s |" % (
                        kind, key, r["steps"], r["batch"], r["torch_seed"], r["train_min"], wall, probe,
                        "complete" if r["complete"] else "INCOMPLETE"))
                else:
                    w("| %s_%s | | | %s | | %s |" % (kind, key, wall, "did not finish" if b is None else e))
            if not wl:
                if "train_min" in r:
                    pr = r["final_probe"] or {}
                    probe = ("mse %.4f" % pr.get("mse_real", pr.get("mse"))) if pr else "—"
                    w("| %s_%s | %d, %d, %d | %.1f min | — | %s | %s |" % (
                        kind, key, r["steps"], r["batch"], r["torch_seed"], r["train_min"], probe,
                        "complete" if r["complete"] else "INCOMPLETE"))
                else:
                    w("| %s_%s | | | — | | not run |" % (kind, key))
    w("| reference_t14_s0 plain-memory test (memory fix) | 300, 8, 42 | 1.2 min | 07:16–07:26 UTC | mse 0.6066 "
      "(step 300) | test only, discarded |")
    w("")
    (OUT / "SCREEN_REPORT.md").write_text("\n".join(L) + "\n" + pathlib.Path("/workspace/runs/stage4_screen/report_tail.md").read_text())
    print("wrote", OUT / "SCREEN_REPORT.md")


if __name__ == "__main__":
    main()
