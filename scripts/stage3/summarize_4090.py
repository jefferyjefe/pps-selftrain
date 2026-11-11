"""Summarize the RTX 4090 rerun (2026-11-11, steps 5 and 6) into results.json and SUMMARY.md in its run directory.
Runs in the openpi venv (numpy, scipy). Safe on partial results. Reuses summarize.py's statistics (Wilson, exact
McNemar, cluster bootstrap) unchanged.

Methods: pi0.5 alone (base), PPS gamma 0.4 (pps), task proxy alone (task). Same 200 paired normal trials and the same
40 saved drop states x 5 branches as the Nov 10 run (RTX PRO 4500, /workspace/runs/eval_t59).

Paired comparisons on the 4090: PPS - pi0.5, task - pi0.5, task - PPS, keyed by trial (normal) or (state, branch) (drop),
clusters = init state (normal) or saved state (drop).
Cross GPU, same method: 4090 - PRO 4500 over the same trial setups (init state, env seed, noise seed). Trajectories
differ from the first policy call on (different GPU numerics), so this compares rates over identical setups. Also the
difference of the PPS - pi0.5 gains between GPUs, with clusters resampled jointly.
"""

import argparse
import glob
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import summarize as SM  # noqa: E402

ROOT = pathlib.Path("/workspace/runs/eval_t59_4090")
OLD = pathlib.Path("/workspace/runs/eval_t59")
METHODS = ("base", "pps", "task")
LABEL = {"base": "pi0.5 alone", "pps": "PPS gamma 0.4", "task": "task proxy alone"}


def cmp(ra, rb, cluster_key):
    """b minus a, paired by key."""
    keys = sorted(set(ra) & set(rb))
    if not keys:
        return None
    a = np.array([ra[k]["success"] for k in keys])
    b = np.array([rb[k]["success"] for k in keys])
    cl = np.array([ra[k][cluster_key] for k in keys])
    m = SM.mcnemar(a, b)
    return {"n_pairs": len(keys), "rate_a": float(a.mean()), "rate_b": float(b.mean()),
            "diff_b_minus_a": float(b.mean() - a.mean()), "cluster_boot95": SM.cluster_boot(a, b, cl),
            "n_clusters": int(len(np.unique(cl))), "a_only": m["base_only_success"], "b_only": m["pps_only_success"],
            "both_success": m["both_success"], "both_failure": m["both_failure"], "mcnemar_p": m["p_exact_two_sided"],
            "same_outcome": int(np.sum(a == b))}


def gain_diff(ob, op, nb, npp, cluster_key, n=10000, seed=0):
    """(pps - base on new GPU) - (pps - base on old GPU), cluster bootstrap over clusters jointly."""
    keys = sorted(set(ob) & set(op) & set(nb) & set(npp))
    if not keys:
        return None
    g_old = np.array([op[k]["success"] for k in keys], float) - np.array([ob[k]["success"] for k in keys], float)
    g_new = np.array([npp[k]["success"] for k in keys], float) - np.array([nb[k]["success"] for k in keys], float)
    cl = np.array([ob[k][cluster_key] for k in keys])
    ids = np.unique(cl)
    d = g_new - g_old
    sums = np.array([d[cl == i].sum() for i in ids])
    cnts = np.array([(cl == i).sum() for i in ids])
    pick = np.random.default_rng(seed).integers(0, len(ids), size=(n, len(ids)))
    boots = sums[pick].sum(1) / cnts[pick].sum(1)
    return {"n": len(keys), "gain_old": float(g_old.mean()), "gain_new": float(g_new.mean()), "diff": float(d.mean()),
            "cluster_boot95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def gpus(recs):
    return sorted({(r.get("server") or {}).get("gpu", "not recorded") for r in recs.values()})


def server_seconds(root, m, kind):
    tot = 0.0
    for f in glob.glob(str(root / m / kind / ("trial_*.npz" if kind == "paired" else "state_*_branch_?.npz"))):
        tot += float(np.load(f)["call_infer_ms"].sum()) / 1000
    return tot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(ROOT))
    p.add_argument("--old", default=str(OLD))
    p.add_argument("--partial-note", default=None)
    args = p.parse_args()
    root, old = pathlib.Path(args.root), pathlib.Path(args.old)
    R = {"generated_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "partial_note": args.partial_note}
    P = {m: SM.load(root / m / "paired.jsonl", max_trial=200) for m in METHODS}
    D = {m: SM.load(root / m / "drop.jsonl") for m in METHODS}
    OP = {m: SM.load(old / m / "paired.jsonl", max_trial=200) for m in ("base", "pps")}
    OD = {m: SM.load(old / m / "drop.jsonl") for m in ("base", "pps")}
    R["gpu_records"] = {m: {"paired": gpus(P[m]), "drop": gpus(D[m])} for m in METHODS}
    R["paired"] = {m: dict(SM.rate(P[m]), **SM.jl_stats(P[m])) for m in METHODS}
    R["drop"] = {m: dict(SM.rate(D[m]), **SM.jl_stats(D[m])) for m in METHODS}
    for m in METHODS:
        ms = [np.load(f)["call_infer_ms"] for f in glob.glob(str(root / m / "drop" / "state_*_branch_?.npz"))]
        ms = [x[1:] if len(x) > 1 else x for x in ms if len(x)]
        if ms:
            R["drop"][m]["mean_infer_ms"] = float(np.mean(np.concatenate(ms)))
        R["paired"][m]["server_compute_s"] = server_seconds(root, m, "paired")
        R["drop"][m]["server_compute_s"] = server_seconds(root, m, "drop")
    R["old_pro4500"] = {"paired": {m: SM.rate(OP[m]) for m in OP}, "drop": {m: SM.rate(OD[m]) for m in OD}}
    R["compare"] = {}
    for a, b in (("base", "pps"), ("base", "task"), ("pps", "task")):
        R["compare"]["%s_vs_%s" % (b, a)] = {"normal": cmp(P[a], P[b], "init_state"), "drop": cmp(D[a], D[b], "state")}
    R["cross_gpu"] = {m: {"normal": cmp(OP[m], P[m], "init_state"), "drop": cmp(OD[m], D[m], "state")}
                      for m in ("base", "pps")}
    R["cross_gpu_gain"] = {"normal": gain_diff(OP["base"], OP["pps"], P["base"], P["pps"], "init_state"),
                           "drop": gain_diff(OD["base"], OD["pps"], D["base"], D["pps"], "state")}
    cn = SM.cosines(sorted(glob.glob(str(root / "pps" / "paired" / "*.npz"))))
    cd = SM.cosines(sorted(glob.glob(str(root / "pps" / "drop" / "state_*_branch_?.npz"))))
    ca = SM.cosines(sorted(glob.glob(str(root / "pps" / "paired" / "*.npz"))) +
                    sorted(glob.glob(str(root / "pps" / "drop" / "state_*_branch_?.npz"))))
    R["cosines_steered_path"] = {"normal": cn, "drop": cd, "pooled": ca}
    R["events"] = {m: {"fired": sum(bool(r.get("event")) for r in P[m].values()), "of": len(P[m])} for m in METHODS}
    for m in METHODS:
        v = list(D[m].values())
        if v:
            R["drop"][m]["details"] = {
                "success_during_drop_frames": sum(r["success_during_drop"] for r in v),
                "bottle_in_grasp_at_handover": sum(bool(r["handover"] and r["handover"]["grasp"]) for r in v),
                "bottle_touching_fingers_at_handover": sum(bool(r["handover"] and r["handover"]["finger"]) for r in v),
                "bottle_on_table_at_handover": sum(bool(r["handover"] and r["handover"]["table"]) for r in v),
                "regrasped": sum(r["regrasp_step"] is not None for r in v),
                "relifted_5cm": sum(r["relift_step"] is not None for r in v),
                "mean_budget_steps": float(np.mean([r["budget_steps"] for r in v]))}
    ck = {}
    for name in ("gamma0_check", "task_only_check"):
        f = root / "checks" / (name + ".json")
        if f.exists():
            j = json.loads(f.read_text())
            ck[name] = {k: j[k] for k in j if k != "rows"}
    f = root / "checks" / "simreplay_check.jsonl"
    if f.exists():
        rr = [json.loads(x) for x in open(f)]
        ck["simreplay"] = {"n": len(rr), "identical": sum(r["identical"] for r in rr),
                           "images_identical": sum(r["restored_images_identical"] for r in rr),
                           "failures": [r for r in rr if not r["identical"]]}
    f = root / "checks" / "replay_check.jsonl"
    if f.exists():
        rr = [json.loads(x) for x in open(f)]
        ck["policy_replay_vs_pro4500"] = {
            "n": len(rr), "fully_identical": sum(r["identical"] for r in rr),
            "identical_before_first_fresh_call": sum(bool(r.get("identical_before_first_fresh_call")) for r in rr),
            "first_diff_at_first_fresh_call": sum(r.get("first_diff_step_any") is not None and
                                                  r.get("first_diff_step_any") == r.get("first_fresh_call_step") for r in rr),
            "images_identical": sum(r["restored_images_identical"] for r in rr),
            "same_outcome_as_pro4500": sum(r["same_outcome"] for r in rr),
            "saved_queue_steps_replayed": int(sum(r["plan_len"] for r in rr))}
    rep = []
    for k in ("state_00_branch_0", "state_01_branch_0", "state_02_branch_0"):
        fa, fb, fm = [root / "base" / "drop" / (k + t + ".npz") for t in ("a", "b", "")]
        if fa.exists() and fb.exists():
            za, zb = np.load(fa), np.load(fb)
            keys = ("step_action", "step_qpos", "step_qvel", "step_obj")
            rep.append({"key": k, "a_eq_b": all(np.array_equal(za[x], zb[x]) for x in keys),
                        "a_eq_main_run": all(np.array_equal(za[x], np.load(fm)[x]) for x in keys) if fm.exists() else None})
    if rep:
        ck["drop_branch_repeat"] = rep
    R["checks"] = ck
    (root / "results.json").write_text(json.dumps(R, indent=2))
    write_md(root, R)
    print(json.dumps({"paired": {m: R["paired"][m]["rate"] for m in METHODS if R["paired"][m]["n"]},
                      "drop": {m: R["drop"][m]["rate"] for m in METHODS if R["drop"][m]["n"]}}, indent=1))


def fr(r):
    return SM.fmt_rate(r)


def fc(c, la, lb):
    if not c:
        return "n/a", "n/a"
    return ("%+.1f pts [%+.1f, %+.1f] (n = %d pairs, %d clusters)" % (
        100 * c["diff_b_minus_a"], 100 * c["cluster_boot95"][0], 100 * c["cluster_boot95"][1], c["n_pairs"], c["n_clusters"]),
        "%.3g (%s only %d, %s only %d)" % (c["mcnemar_p"], lb, c["b_only"], la, c["a_only"]))


def write_md(root, R):
    L = ["# t59 evaluation on the RTX 4090: pi0.5 alone, PPS gamma 0.4, task proxy alone\n",
         "Generated %s UTC by scripts/stage3/summarize_4090.py from the files in %s. All policy calls on: %s.\n" % (
             R["generated_utc"], root, ", ".join(sorted({g for m in R["gpu_records"].values() for v in m.values() for g in v})))]
    if R["partial_note"]:
        L.append("**PARTIAL RESULTS.** %s\n" % R["partial_note"])
    L += ["## Success rates (Wilson 95%)\n", "| Test | pi0.5 alone | PPS gamma 0.4 | task proxy alone |", "| --- | --- | --- | --- |"]
    for name, key in (("Normal, 200 paired trials", "paired"), ("Mid-carry drop, 40 states x 5 branches", "drop")):
        L.append("| %s | %s | %s | %s |" % (name, fr(R[key]["base"]), fr(R[key]["pps"]), fr(R[key]["task"])))
    L += ["\n## Paired differences (4090 only)\n",
          "| Comparison | Test | difference, 95% cluster bootstrap | McNemar exact p (discordant pairs) | same outcome |",
          "| --- | --- | --- | --- | --- |"]
    for key, la, lb in (("pps_vs_base", "pi0.5", "PPS"), ("task_vs_base", "pi0.5", "task"), ("task_vs_pps", "PPS", "task")):
        for t in ("normal", "drop"):
            c = R["compare"][key][t]
            d, pv = fc(c, la, lb)
            L.append("| %s minus %s | %s | %s | %s | %s |" % (lb, la, t, d, pv, "%d / %d" % (c["same_outcome"], c["n_pairs"]) if c else "n/a"))
    L += ["\n## RTX 4090 vs RTX PRO 4500 (Nov 10), same method, same 200 trial setups / 200 drop branches\n",
          "Trajectories differ from the first policy call (GPU numerics), so pairs share only the setup (init state, env "
          "seed, noise stream) or the saved state and branch noise.\n",
          "| Method | Test | PRO 4500 | RTX 4090 | 4090 minus PRO 4500, cluster bootstrap | McNemar p (4090 only, PRO 4500 only) | same outcome |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for m in ("base", "pps"):
        for t, key in (("normal", "paired"), ("drop", "drop")):
            c = R["cross_gpu"][m][t]
            d, pv = fc(c, "PRO 4500", "4090")
            L.append("| %s | %s | %s | %s | %s | %s | %s |" % (LABEL[m], t, fr(R["old_pro4500"][key][m]), fr(R[key][m]), d, pv,
                                                            "%d / %d" % (c["same_outcome"], c["n_pairs"]) if c else "n/a"))
    for t in ("normal", "drop"):
        g = R["cross_gpu_gain"][t]
        if g:
            L.append("\nPPS minus pi0.5 gain, %s: %+.1f pts on the PRO 4500, %+.1f pts on the 4090; difference %+.1f pts "
                     "[%+.1f, %+.1f] (cluster bootstrap, n = %d)." % (t, 100 * g["gain_old"], 100 * g["gain_new"], 100 * g["diff"],
                                                                      100 * g["cluster_boot95"][0], 100 * g["cluster_boot95"][1], g["n"]))
    c = R["cosines_steered_path"]
    if c.get("pooled"):
        L += ["\n## Velocity cosines along PPS's steered path (4090)\n", "| Pair | dims | normal | drop | pooled |", "| --- | --- | --- | --- | --- |"]
        for pair in ("ref_base", "task_ref", "task_base"):
            for dd in ("first7", "all32"):
                k = "%s_%s" % (pair, dd)
                L.append("| %s | %s | %s |" % (pair.replace("_", " vs "), dd, " | ".join(
                    "%.4f" % c[s][k] if c.get(s) else "n/a" for s in ("normal", "drop", "pooled"))))
        L.append("\nMean over %d PPS calls x 10 steps. mean |v_base| %.3f, mean |gamma (v_task - v_ref)| %.3f." % (
            c["pooled"]["n_calls"], c["pooled"]["norm_vbase_all32"], c["pooled"]["norm_delta_all32"]))
    L.append("\n## Checks (all on the 4090)\n")
    ck = R["checks"]
    if "gamma0_check" in ck:
        g = ck["gamma0_check"]
        L.append("- gamma = 0: PPS code path bit exact with pi0.5 (stock eager, SteerablePolicy, raw 32-dim output) on %d "
                 "observations: **%s**; gamma 0.4 changes actions on all: %s. Median ms per call: pi0.5 %.0f, PPS %.0f." % (
                     g["n_observations"], g["bit_exact_all"], g["pps04_changes_actions_all"], g["median_ms"]["base"], g["median_ms"]["pps"]))
    if "task_only_check" in ck:
        g = ck["task_only_check"]
        L.append("- Task proxy alone: fast path (proxy Euler loop, base not run) bit exact with the reference path "
                 "(pps_real.sampler, base velocity replaced by v_task) on %d observations: **%s**. Median ms: fast %.0f, "
                 "reference %.0f." % (g["n_observations"], g["bit_exact_all"], g["median_ms"]["fast"], g["median_ms"]["hook"]))
    if "simreplay" in ck:
        s = ck["simreplay"]
        L.append("- Saved states, sim side: restore + open loop replay of the Nov 10 logged actions reproduced qpos, qvel, "
                 "bottle pose and outcome bit for bit in **%d of %d** states (restored images identical in %d)." % (
                     s["identical"], s["n"], s["images_identical"]))
    if "policy_replay_vs_pro4500" in ck:
        s = ck["policy_replay_vs_pro4500"]
        L.append("- Saved states, policy in the loop (the Nov 10 check): fully identical to the PRO 4500 trajectory in %d of %d; "
                 "identical through the saved action queue (%d steps in total) up to the first fresh 4090 policy call in "
                 "**%d of %d**; first difference exactly at that call in %d; same outcome as on the PRO 4500 in %d." % (
                     s["fully_identical"], s["n"], s["saved_queue_steps_replayed"], s["identical_before_first_fresh_call"], s["n"],
                     s["first_diff_at_first_fresh_call"], s["same_outcome_as_pro4500"]))
    if "drop_branch_repeat" in ck:
        rr = ck["drop_branch_repeat"]
        L.append("- Drop branch determinism: %d (state, branch) pairs run twice on different server processes, identical in %d; "
                 "identical to the main drop run in %d of %d." % (len(rr), sum(x["a_eq_b"] for x in rr),
                                                                sum(bool(x["a_eq_main_run"]) for x in rr),
                                                                sum(x["a_eq_main_run"] is not None for x in rr)))
    L.append("\n## Details\n")
    for m in METHODS:
        for key in ("paired", "drop"):
            d = R[key][m]
            if d.get("n"):
                L.append("- %s, %s: mean %.1f policy calls, %.0f ms per call (server), %.1f s wall per trial, %.0f s server "
                         "compute in total (RTX 4090), errors: %s" % (
                             LABEL[m], "normal" if key == "paired" else "drop", d["mean_policy_calls"], d["mean_infer_ms"] or float("nan"),
                             d["mean_wall_s"], d["server_compute_s"], d["errors"] or "none"))
    L.append("- Mid-carry event fired in: %s." % ", ".join("%s %d of %d" % (LABEL[m], v["fired"], v["of"]) for m, v in R["events"].items()))
    for m in METHODS:
        d = R["drop"][m].get("details")
        if d:
            L.append("- Drop, %s: out of grasp at handover %d of %d, on the table %d; succeeded during drop frames %d; "
                     "re-grasped %d, re-lifted 5 cm %d; mean remaining budget %.0f steps." % (
                         LABEL[m], R["drop"][m]["n"] - d["bottle_in_grasp_at_handover"], R["drop"][m]["n"],
                         d["bottle_on_table_at_handover"], d["success_during_drop_frames"], d["regrasped"], d["relifted_5cm"],
                         d["mean_budget_steps"]))
    notes = root / "SUMMARY_NOTES.md"
    if notes.exists():
        L += ["", notes.read_text()]
    (root / "SUMMARY.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
