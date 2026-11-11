"""Summarize the stage 3 t59 evaluation into results.json and SUMMARY.md. Runs in the openpi venv (numpy, scipy).

Safe on partial results (every trial is appended to its JSONL as it finishes). Only keys finished by both methods
enter the paired tests.

Statistics
- Success rate per method and test with a Wilson 95% interval over the finished trials.
- Paired test: exact McNemar (two sided binomial test on the discordant pairs, p = 0.5).
- Paired difference (PPS minus pi0.5) with a 95% cluster bootstrap interval (10,000 resamples of clusters). Clusters
  are LIBERO init states for the normal trials (4 seeds each) and saved states for the drop test (5 branches each),
  because trials in a cluster share a start and are not independent. Wilson and McNemar treat trials as independent.
- Velocity cosines along PPS's steered path: every PPS policy call logs, at each of the 10 denoising steps, the cosine
  over the flattened chunk (first 7 dims, all 32) between v_ref and v_base and between v_task and v_ref. Reported as
  the mean over calls and steps (as check_task_paper.py and the paper's Fig. 10b), per test and pooled.
"""

import argparse
import glob
import json
import math
import pathlib
import time

import numpy as np
from scipy import stats

ROOT = pathlib.Path("/workspace/runs/eval_t59")
METHODS = ("base", "pps")
LABEL = {"base": "pi0.5 alone", "pps": "PPS gamma 0.4"}


def load(path, max_trial=None):
    out = {}
    if path.exists():
        for line in open(path):
            r = json.loads(line)
            if max_trial is not None and r["trial"] >= max_trial:
                continue  # extra pi0.5 trials run only to fill the drop pool
            out.setdefault(r["key"], r)
    return out


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def mcnemar(a, b):
    """a, b boolean arrays (base, pps) over the same keys. Exact two sided."""
    n01 = int(np.sum(a & ~b))  # base success, pps failure
    n10 = int(np.sum(~a & b))  # pps success, base failure
    n = n01 + n10
    p = float(stats.binomtest(n10, n, 0.5).pvalue) if n else 1.0
    return {"base_only_success": n01, "pps_only_success": n10, "both_success": int(np.sum(a & b)),
            "both_failure": int(np.sum(~a & ~b)), "discordant": n, "p_exact_two_sided": p}


def cluster_boot(a, b, cl, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.unique(cl)
    idx = [np.nonzero(cl == i)[0] for i in ids]
    d = b.astype(float) - a.astype(float)
    sums = np.array([d[ix].sum() for ix in idx])
    cnts = np.array([len(ix) for ix in idx])
    pick = rng.integers(0, len(ids), size=(n, len(ids)))
    boots = sums[pick].sum(1) / cnts[pick].sum(1)
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def rate(recs):
    k = sum(r["success"] for r in recs.values())
    n = len(recs)
    lo, hi = wilson(k, n)
    return {"n": n, "successes": k, "rate": k / n if n else float("nan"), "wilson95": [lo, hi]}


def compare(ra, rb, cluster_key):
    keys = sorted(set(ra) & set(rb))
    if not keys:
        return None
    a = np.array([ra[k]["success"] for k in keys])
    b = np.array([rb[k]["success"] for k in keys])
    cl = np.array([ra[k][cluster_key] for k in keys])
    out = {"n_pairs": len(keys), "base_rate": float(a.mean()), "pps_rate": float(b.mean()),
           "diff_pps_minus_base": float(b.mean() - a.mean()), "mcnemar": mcnemar(a, b),
           "diff_cluster_bootstrap95": cluster_boot(a, b, cl), "cluster": cluster_key,
           "n_clusters": int(len(np.unique(cl)))}
    return out


def cosines(files):
    keys = ("ref_base_first7", "ref_base_all32", "task_ref_first7", "task_ref_all32", "task_base_first7",
            "task_base_all32", "norm_vbase_all32", "norm_delta_all32")
    acc = {k: [] for k in keys}
    for f in files:
        z = np.load(f)
        if "pps_ref_base_all32" not in z.files:
            continue
        for k in keys:
            acc[k].append(z["pps_" + k])
    if not acc[keys[0]]:
        return None
    out = {"n_calls": int(sum(len(x) for x in acc[keys[0]]))}
    for k in keys:
        x = np.concatenate(acc[k])  # (calls, 10)
        out[k] = float(x.mean())
        out[k + "_per_step"] = [float(v) for v in x.mean(0)]
    return out


def jl_stats(recs):
    if not recs:
        return {}
    v = list(recs.values())
    ms = [r["infer_ms_mean"] for r in v if r.get("infer_ms_mean")]
    return {"errors": [r["key"] for r in v if r.get("error")], "mean_wall_s": float(np.mean([r["wall_s"] for r in v])),
            "mean_policy_calls": float(np.mean([r["policy_calls"] for r in v])),
            "mean_infer_ms": float(np.mean(ms)) if ms else None,
            "mean_env_steps": float(np.mean([r["env_steps"] for r in v]))}


def fmt_rate(r):
    if not r or not r["n"]:
        return "no trials"
    return "%d / %d = %.1f%% [%.1f, %.1f]" % (r["successes"], r["n"], 100 * r["rate"], 100 * r["wilson95"][0],
                                             100 * r["wilson95"][1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(ROOT))
    p.add_argument("--partial-note", default=None)
    args = p.parse_args()
    root = pathlib.Path(args.root)
    R = {"generated_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "partial_note": args.partial_note}
    paired = {m: load(root / m / "paired.jsonl", max_trial=200) for m in METHODS}
    drop = {m: load(root / m / "drop.jsonl") for m in METHODS}
    R["paired"] = {m: dict(rate(paired[m]), **jl_stats(paired[m])) for m in METHODS}
    R["drop"] = {m: dict(rate(drop[m]), **jl_stats(drop[m])) for m in METHODS}
    for m in METHODS:  # drop records do not store per call time; read it from the per trial npz files
        ms = [np.load(f)["call_infer_ms"] for f in glob.glob(str(root / m / "drop" / "state_*_branch_?.npz"))]
        ms = [x[1:] if len(x) > 1 else x for x in ms if len(x)]
        if ms:
            R["drop"][m]["mean_infer_ms"] = float(np.mean(np.concatenate(ms)))
    R["paired_test_normal"] = compare(paired["base"], paired["pps"], "init_state")
    R["paired_test_drop"] = compare(drop["base"], drop["pps"], "state")
    cos_n = cosines(sorted(glob.glob(str(root / "pps" / "paired" / "*.npz"))))  # PPS has no extra trials
    cos_d = cosines(sorted(glob.glob(str(root / "pps" / "drop" / "state_*_branch_?.npz"))))
    cos_all = cosines(sorted(glob.glob(str(root / "pps" / "paired" / "*.npz"))) +
                      sorted(glob.glob(str(root / "pps" / "drop" / "state_*_branch_?.npz"))))
    R["cosines_steered_path"] = {"normal": cos_n, "drop": cos_d, "pooled": cos_all,
                                 "paper_fig10b": {"ref_base_steered": 0.946, "task_ref": 0.865}}
    # events and pool
    ev = [r for r in paired["base"].values() if r.get("event")]
    R["mid_carry_events_base"] = {"fired": len(ev), "of": len(paired["base"])}
    ev_p = [r for r in paired["pps"].values() if r.get("event")]
    R["mid_carry_events_pps"] = {"fired": len(ev_p), "of": len(paired["pps"])}
    man_p = root / "pool" / "manifest.json"
    R["pool"] = json.loads(man_p.read_text()) if man_p.exists() else None
    if R["pool"]:
        R["pool"]["source_budgets"] = [409 - s["event_step"] for s in R["pool"]["states"]]
        R["pool"] = {k: v for k, v in R["pool"].items() if k != "states"} | {
            "trials": [s["trial"] for s in R["pool"]["states"]],
            "event_steps": [s["event_step"] for s in R["pool"]["states"]],
            "plan_len_at_event": [s["plan_len"] for s in R["pool"]["states"]]}
    # drop details
    for m in METHODS:
        v = list(drop[m].values())
        if v:
            R["drop"][m]["details"] = {
                "success_during_drop_frames": sum(r["success_during_drop"] for r in v),
                "bottle_touching_fingers_at_handover": sum(bool(r["handover"] and r["handover"]["finger"]) for r in v),
                "bottle_in_grasp_at_handover": sum(bool(r["handover"] and r["handover"]["grasp"]) for r in v),
                "bottle_on_table_at_handover": sum(bool(r["handover"] and r["handover"]["table"]) for r in v),
                "regrasped": sum(r["regrasp_step"] is not None for r in v),
                "relifted_5cm": sum(r["relift_step"] is not None for r in v),
                "mean_budget_steps": float(np.mean([r["budget_steps"] for r in v]))}
    # checks
    ck = {}
    g = root / "checks" / "gamma0_check.json"
    if g.exists():
        gj = json.loads(g.read_text())
        ck["gamma0"] = {k: gj[k] for k in ("bit_exact_all", "n_observations", "pps04_changes_actions_all", "median_ms")}
    rp = root / "checks" / "replay_check.jsonl"
    if rp.exists():
        rr = [json.loads(x) for x in open(rp)]
        ck["replay"] = {"n": len(rr), "identical": sum(r["identical"] for r in rr),
                        "images_identical": sum(r["restored_images_identical"] for r in rr),
                        "failures": [r for r in rr if not r["identical"]]}
    rep = {}
    for m in METHODS:
        for tag in ("a", "b"):
            rep[tag] = load(root / m / ("repeat%s.jsonl" % tag)) if not rep.get(tag) else rep[tag]
    if rep.get("a") and rep.get("b"):
        res = []
        for k in sorted(set(rep["a"]) & set(rep["b"])):
            za = np.load(root / "base" / "drop" / (k + "a.npz"))
            zb = np.load(root / "base" / "drop" / (k + "b.npz"))
            same = all(np.array_equal(za[x], zb[x]) for x in ("step_action", "step_qpos", "step_qvel"))
            zm = root / "base" / "drop" / (k + ".npz")
            same_main = None
            if zm.exists():
                zz = np.load(zm)
                same_main = all(np.array_equal(za[x], zz[x]) for x in ("step_action", "step_qpos", "step_qvel"))
            res.append({"key": k, "a_eq_b": bool(same), "a_eq_main_run": same_main,
                        "ports": [rep["a"][k]["server"].get("port"), rep["b"][k]["server"].get("port")]})
        ck["drop_branch_repeat"] = res
    s2 = pathlib.Path("/workspace/pps-selftrain/results/stage2/perturb/none.jsonl")
    if s2.exists():
        st2 = {json.loads(x)["episode"]: json.loads(x) for x in open(s2)}
        both = [t for t in st2 if "trial_%04d" % t in paired["base"]]
        if both:
            ck["stage2_base_episodes_0_29"] = {
                "n": len(both),
                "same_outcome": sum(st2[t]["success"] == paired["base"]["trial_%04d" % t]["success"] for t in both),
                "same_outcome_and_steps": sum(st2[t]["success"] == paired["base"]["trial_%04d" % t]["success"] and
                                              st2[t]["env_steps"] == paired["base"]["trial_%04d" % t]["env_steps"] for t in both),
                "stage2_successes": sum(st2[t]["success"] for t in both),
                "now_successes": sum(paired["base"]["trial_%04d" % t]["success"] for t in both)}
    R["checks"] = ck
    (root / "results.json").write_text(json.dumps(R, indent=2))
    write_md(root, R)
    print(json.dumps({k: R[k] for k in ("paired_test_normal", "paired_test_drop")}, indent=1))


def write_md(root, R):
    L = []
    L.append("# t59 evaluation: pi0.5 alone vs PPS at gamma 0.4\n")
    L.append("Generated %s UTC by scripts/stage3/summarize.py from the files in /workspace/runs/eval_t59.\n" % R["generated_utc"])
    if R["partial_note"]:
        L.append("**PARTIAL RESULTS.** %s Counts below are the trials finished before that.\n" % R["partial_note"])
    L.append("## Results\n")
    L.append("| Test | pi0.5 alone | PPS gamma 0.4 | PPS minus pi0.5, paired, 95% cluster bootstrap | McNemar exact p |")
    L.append("| --- | --- | --- | --- | --- |")
    for name, key, tkey in (("Normal, paired trials", "paired", "paired_test_normal"),
                            ("Mid-carry drop", "drop", "paired_test_drop")):
        t = R[tkey]
        diff = "%+.1f pts [%+.1f, %+.1f] (n = %d pairs, %d clusters)" % (
            100 * t["diff_pps_minus_base"], 100 * t["diff_cluster_bootstrap95"][0], 100 * t["diff_cluster_bootstrap95"][1],
            t["n_pairs"], t["n_clusters"]) if t else "n/a"
        pv = "%.3g (discordant: PPS only %d, pi0.5 only %d)" % (
            t["mcnemar"]["p_exact_two_sided"], t["mcnemar"]["pps_only_success"], t["mcnemar"]["base_only_success"]) if t else "n/a"
        L.append("| %s | %s | %s | %s | %s |" % (name, fmt_rate(R[key]["base"]), fmt_rate(R[key]["pps"]), diff, pv))
    L.append("\nSuccess rates are successes / trials with a Wilson 95% interval in brackets.\n")
    c = R["cosines_steered_path"]
    L.append("## Velocity cosines along the steered path (PPS rollouts)\n")
    L.append("| Pair | dims | normal trials | drop trials | pooled | paper (Fig. 10b) |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    for pair, paper in (("ref_base", 0.946), ("task_ref", 0.865), ("task_base", None)):
        for d in ("first7", "all32"):
            k = "%s_%s" % (pair, d)
            vals = ["%.4f" % c[s][k] if c.get(s) else "n/a" for s in ("normal", "drop", "pooled")]
            L.append("| %s | %s | %s | %s | %s | %s |" % (pair.replace("_", " vs "), d, *vals, paper if paper else ""))
    if c.get("pooled"):
        L.append("\nMean over %d PPS policy calls x 10 denoising steps. Per step (pooled, steps at t = 1.0 ... 0.1):\n" % c["pooled"]["n_calls"])
        for k in ("ref_base_first7", "ref_base_all32", "task_ref_first7", "task_ref_all32"):
            L.append("- %s: %s" % (k, ", ".join("%.3f" % v for v in c["pooled"][k + "_per_step"])))
        L.append("- mean |v_base| %.3f, mean |gamma (v_task - v_ref)| %.3f (all 32 dims, per step)\n" % (
            c["pooled"]["norm_vbase_all32"], c["pooled"]["norm_delta_all32"]))
    L.append("## Safety checks\n")
    ck = R["checks"]
    if "gamma0" in ck:
        g = ck["gamma0"]
        L.append("- gamma = 0: PPS code path bit exact with pi0.5 on %d observations: **%s**. gamma 0.4 changes actions on all: %s. "
                 "Median ms per call: pi0.5 %.0f, PPS %.0f." % (g["n_observations"], g["bit_exact_all"],
                                                               g["pps04_changes_actions_all"], g["median_ms"]["base"], g["median_ms"]["pps"]))
    if "replay" in ck:
        r = ck["replay"]
        L.append("- Branching: restoring a saved state with its action queue and noise generator and replaying pi0.5 reproduced "
                 "the original trajectory bit for bit (actions, qpos, qvel, bottle pose, outcome) in **%d of %d** states "
                 "(restored observation images identical in %d)." % (r["identical"], r["n"], r["images_identical"]))
    if "drop_branch_repeat" in ck:
        rr = ck["drop_branch_repeat"]
        L.append("- Drop branch determinism: %d (state, branch) pairs run twice on different server processes, identical in %d; "
                 "identical to the main drop run in %d of %d." % (
                     len(rr), sum(x["a_eq_b"] for x in rr), sum(bool(x["a_eq_main_run"]) for x in rr),
                     sum(x["a_eq_main_run"] is not None for x in rr)))
    if "stage2_base_episodes_0_29" in ck:
        x = ck["stage2_base_episodes_0_29"]
        L.append("- Cross check with stage 2: pi0.5 trials 0-29 use the same init states, env seeds and noise as stage 2's eager "
                 "base episodes 0-29 (run on an RTX PRO 6000; this pod has an RTX PRO 4500). Same outcome in %d of %d, same "
                 "outcome and step count in %d; successes %d then, %d now." % (
                     x["same_outcome"], x["n"], x["same_outcome_and_steps"], x["stage2_successes"], x["now_successes"]))
    L.append("")
    L.append("## Details\n")
    for m in METHODS:
        for key in ("paired", "drop"):
            d = R[key][m]
            if d.get("n"):
                L.append("- %s, %s: mean %.1f policy calls, %.0f ms per call (server), %.1f s wall per trial, errors: %s" % (
                    LABEL[m], "normal" if key == "paired" else "drop", d["mean_policy_calls"], d["mean_infer_ms"] or float("nan"),
                    d["mean_wall_s"], d["errors"] or "none"))
    e = R["mid_carry_events_base"]
    L.append("- Mid-carry event fired in %d of %d pi0.5 normal trials (%d of %d for PPS)." % (
        e["fired"], e["of"], R["mid_carry_events_pps"]["fired"], R["mid_carry_events_pps"]["of"]))
    if R["pool"]:
        L.append("- Drop pool: %d states from pi0.5 trials %s (finished prefix 0..%d), minimum remaining budget %s steps. "
                 "Event steps %s; remaining budget per state %s." % (
            len(R["pool"]["trials"]), R["pool"]["trials"], R["pool"]["prefix_trials"] - 1, R["pool"].get("min_budget"),
            R["pool"]["event_steps"], R["pool"]["source_budgets"]))
    for m in METHODS:
        d = R["drop"][m].get("details")
        if d:
            L.append("- Drop, %s: bottle out of the fingers' grasp at handover in %d of %d, touching a finger %d, on the table %d; "
                     "succeeded during the drop frames %d; re-grasped %d, re-lifted 5 cm %d; mean remaining budget %.0f steps." % (
                         LABEL[m], R["drop"][m]["n"] - d["bottle_in_grasp_at_handover"], R["drop"][m]["n"],
                         d["bottle_touching_fingers_at_handover"], d["bottle_on_table_at_handover"],
                         d["success_during_drop_frames"], d["regrasped"], d["relifted_5cm"], d["mean_budget_steps"]))
    notes = root / "SUMMARY_NOTES.md"
    if notes.exists():
        L.append("")
        L.append(notes.read_text())
    (root / "SUMMARY.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
