"""Summary tables, statistics and the trade-off plot for the faithful t59 evaluation. openpi venv (CPU only).

Reads <root>/<cell>/{paired,drop,hard}.jsonl and the per-episode npz step logs; pi0.5's normal trials come from
/workspace/runs/eval_t59_4090/base when <root>/repro_check/REUSE_OK exists (else from <root>/base), and the v3 ours
normal trials (faithful vs v3 only) from /workspace/runs/eval_t59_4090/{pps,task} when <root>/checks/V3_REUSE_OK
exists (compare_v3.py).
Writes <root>/summary.json, <root>/TABLES.md, <root>/tradeoff.png (x = normal success, y = drop-bank recovery) and
<root>/tradeoff_drop_success.png (y = drop-bank task completion).

Statistics: Wilson 95% intervals; paired differences vs pi0.5 on the common keys with a cluster bootstrap (10,000
resamples, clusters = init state for normal trials, start for harder trials, bank state for drop branches); exact
McNemar p on the discordant pairs. Recovery: after the 10 drop steps, 25 consecutive env steps with both fingers on the
bottle, gripper commanded closed (a[6] > 0) and no bottle-table contact.
"""

import argparse
import json
import pathlib

import numpy as np
from scipy.stats import binom

ROOT = pathlib.Path("/workspace/runs/eval_t59_faithful")
NOV11 = pathlib.Path("/workspace/runs/eval_t59_4090")
BASE_NOV11 = NOV11 / "base"
HOLD, DROP_FRAMES = 25, 10
PPS = ["pps_f_g02", "pps_f_g04", "pps_f_g06", "pps_f_g08", "pps_f_g10"]
BLEND = ["blend_f_g02", "blend_f_g04", "blend_f_g06", "blend_f_g08"]
ITEM1 = ["pps_f_g04", "task_f"]
V3 = ["pps_v3_g04", "task_v3"]
NAMES = {"base": "π0.5", "task_f": "task proxy alone (faithful)", "pps_v3_g04": "PPS γ 0.4 (v3 ours)",
         "task_v3": "task proxy alone (v3 ours)"}
for c in PPS:
    NAMES[c] = "PPS γ %.1f (faithful)" % (int(c[-2:]) / 10)
for c in BLEND:
    NAMES[c] = "blend γ %.1f (faithful)" % (int(c[-2:]) / 10)
RNG = np.random.default_rng(0)


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar(x, y):
    b, c = int(np.sum((x == 1) & (y == 0))), int(np.sum((x == 0) & (y == 1)))
    p = 1.0 if b + c == 0 else min(1.0, 2 * binom.cdf(min(b, c), b + c, 0.5))
    return p, b, c


def cluster_boot(x, y, cl, n=10000):
    u, inv = np.unique(cl, return_inverse=True)
    dx = np.bincount(inv, weights=x - y, minlength=len(u))
    cnt = np.bincount(inv, minlength=len(u))
    idx = RNG.integers(0, len(u), (n, len(u)))
    est = dx[idx].sum(1) / cnt[idx].sum(1)
    return float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def load_jsonl(p):
    out = {}
    if p.exists():
        for line in open(p):
            r = json.loads(line)
            out[r["key"]] = r
    return out


def recovered(npz_path):
    z = np.load(npz_path)
    g, a, tb = z["step_grasp"][DROP_FRAMES:], z["step_a6"][DROP_FRAMES:] > 0, z["step_table"][DROP_FRAMES:]
    ok = g.astype(bool) & a & ~tb.astype(bool)
    run = 0
    for i, v in enumerate(ok):
        run = run + 1 if v else 0
        if run >= HOLD:
            return True, int(i - HOLD + 1)
    return False, None


V3_NOV11 = {"pps_v3_g04": "pps", "task_v3": "task"}


def load_cell(root, cell, reuse_base, reuse_v3=False):
    d = {}
    if cell == "base" and reuse_base:
        pj = BASE_NOV11 / "paired.jsonl"
    elif cell in V3_NOV11 and reuse_v3:
        pj = NOV11 / V3_NOV11[cell] / "paired.jsonl"  # reproduced bit for bit (compare_v3.py)
    else:
        pj = root / cell / "paired.jsonl"
    d["normal"] = {r["trial"]: r for r in load_jsonl(pj).values()}
    d["hard"] = {r["trial"]: r for r in load_jsonl(root / cell / "hard.jsonl").values()}
    dr = {}
    for r in load_jsonl(root / cell / "drop.jsonl").values():
        rec, at = recovered(root / cell / "drop" / ("state_%02d_branch_%d.npz" % (r["state"], r["branch"])))
        r["recovered"], r["recovered_at"] = rec, at
        dr[(r["state"], r["branch"])] = r
    d["drop"] = dr
    return d


def pad_audit(root, cells):
    viol, checked, n_ep = 0, 0, 0
    for c in cells:
        for sub in ("paired", "drop", "hard"):
            for f in (root / c / sub).glob("*.npz") if (root / c / sub).exists() else []:
                z = np.load(f)
                if "call_pad_violations" in z and len(z["call_pad_violations"]):
                    v = z["call_pad_violations"]
                    viol = max(viol, int(v.max()))
                    checked += int(len(v))
                    n_ep += 1
    return {"max_server_violation_counter": viol, "calls_with_counter": checked, "episodes": n_ep}


def cosine_check(root, n=20):
    vals, per_trial = [], []
    for t in range(n):
        f = root / "pps_f_g04" / "paired" / ("trial_%04d.npz" % t)
        if not f.exists():
            return None
        z = np.load(f)
        v = z["pps_ref_base_real"]
        vals.append(v.ravel())
        per_trial.append(float(np.mean(v)))
    allv = np.concatenate(vals)
    return {"n_trials": n, "n_points": int(len(allv)), "mean_all_points": float(np.mean(allv)),
            "mean_of_trial_means": float(np.mean(per_trial)), "min_trial_mean": float(np.min(per_trial)),
            "by_euler_step": np.mean(np.concatenate([np.load(root / "pps_f_g04" / "paired" / ("trial_%04d.npz" % t))
                                                     ["pps_ref_base_real"] for t in range(n)]), 0).tolist()}


def row(test, cell, xs, keys, ys=None, cl=None, metric="success"):
    k, n = int(np.sum(xs)), len(xs)
    lo, hi = wilson(k, n)
    r = {"test": test, "metric": metric, "cell": cell, "name": NAMES.get(cell, cell), "k": k, "n": n,
         "rate": k / n if n else float("nan"), "ci": [lo, hi]}
    if ys is not None and cell != "base":
        r["diff"] = float(np.mean(xs) - np.mean(ys))
        r["diff_ci"] = cluster_boot(xs.astype(float), ys.astype(float), cl)
        r["mcnemar_p"], r["only_cell"], r["only_base"] = mcnemar(xs, ys)
    return r


def paired_arrays(a, b, keys, field):
    return (np.array([int(bool(a[k][field])) for k in keys]), np.array([int(bool(b[k][field])) for k in keys]))


def fmt(r):
    s = "%d/%d = %.1f%% [%.1f, %.1f]" % (r["k"], r["n"], 100 * r["rate"], 100 * r["ci"][0], 100 * r["ci"][1])
    if "diff" in r:
        d = "%+.1f [%+.1f, %+.1f]" % (100 * r["diff"], 100 * r["diff_ci"][0], 100 * r["diff_ci"][1])
        m = "%.2g (%d / %d)" % (r["mcnemar_p"], r["only_cell"], r["only_base"])
    else:
        d = m = "—"
    return s, d, m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(ROOT))
    args = p.parse_args()
    root = pathlib.Path(args.root)
    reuse = (root / "repro_check" / "REUSE_OK").exists()
    cells = ["base"] + PPS + BLEND + ["task_f"] + V3
    v3_reuse = (root / "checks" / "V3_REUSE_OK").exists()
    D = {c: load_cell(root, c, reuse, v3_reuse) for c in cells}
    man = json.loads((root / "bank" / "manifest.json").read_text())
    src_of = {s["state"]: s["source"] for s in man["states"]}
    rows = []
    base = D["base"]
    # normal trials: item-1 cells on all common trials, sweep cells on theirs (0-99)
    for c in cells:
        if c in V3:
            continue
        keys = sorted(set(D[c]["normal"]) & set(base["normal"]))
        if not keys:
            continue
        x, y = paired_arrays(D[c]["normal"], base["normal"], keys, "success")
        rows.append(row("normal", c, x, keys, y, np.array(keys) % 50))
    for c in cells:
        keys = sorted(set(D[c]["drop"]) & set(base["drop"]))
        if not keys:
            continue
        for metric, field in (("success", "success"), ("recovery", "recovered")):
            x, y = paired_arrays(D[c]["drop"], base["drop"], keys, field)
            rows.append(row("drop", c, x, keys, y, np.array([k[0] for k in keys]), metric))
    for c in cells:
        keys = sorted(set(D[c]["hard"]) & set(base["hard"]))
        if not keys:
            continue
        x, y = paired_arrays(D[c]["hard"], base["hard"], keys, "success")
        rows.append(row("hard", c, x, keys, y, np.array(keys)))

    # retained / lost / acquired on the bank, by source
    def rla(a, b, field, by_src=True):
        keys = sorted(set(a) & set(b))
        out = {}
        for src in ["all"] + [s["source"] for s in man["sources"]]:
            ks = [k for k in keys if src == "all" or src_of[k[0]] == src]
            x = np.array([bool(a[k][field]) for k in ks])
            y = np.array([bool(b[k][field]) for k in ks])
            out[src] = {"n": len(ks), "retained": int(np.sum(x & y)), "lost": int(np.sum(~x & y)),
                        "acquired": int(np.sum(x & ~y)), "neither": int(np.sum(~x & ~y))}
        return out
    rla_tab = {}
    for c in cells:
        if c == "base" or not D[c]["drop"]:
            continue
        rla_tab[c + "_vs_base"] = {f: rla(D[c]["drop"], base["drop"], fl) for f, fl in
                                   (("success", "success"), ("recovery", "recovered"))}
    for p_, t_ in (("pps_f_g04", "task_f"), ("pps_v3_g04", "task_v3")):
        if D[p_]["drop"] and D[t_]["drop"]:
            rla_tab[p_ + "_vs_" + t_] = {f: rla(D[p_]["drop"], D[t_]["drop"], fl) for f, fl in
                                         (("success", "success"), ("recovery", "recovered"))}
    # per-source rates per cell
    by_src = {}
    for c in cells:
        if not D[c]["drop"]:
            continue
        by_src[c] = {}
        for src in [s["source"] for s in man["sources"]]:
            ks = [k for k in D[c]["drop"] if src_of[k[0]] == src]
            by_src[c][src] = {"n": len(ks), "success": float(np.mean([D[c]["drop"][k]["success"] for k in ks])) if ks else None,
                              "recovery": float(np.mean([D[c]["drop"][k]["recovered"] for k in ks])) if ks else None}

    # faithful vs v3 on shared cells: normal trials (v3 from the Nov 11 4090 run, if reproduced here) and the bank
    fv = []
    if v3_reuse:
        for f_, v_ in (("pps_f_g04", "pps_v3_g04"), ("task_f", "task_v3")):
            vn = D[v_]["normal"]
            keys = sorted(set(D[f_]["normal"]) & set(vn))
            if not keys:
                continue
            x, y = paired_arrays(D[f_]["normal"], vn, keys, "success")
            r = row("normal", f_, x, keys, y, np.array(keys) % 50, "success")
            r["vs"] = v_
            r["v3_k"] = int(y.sum())
            fv.append(r)
    for f_, v_ in (("pps_f_g04", "pps_v3_g04"), ("task_f", "task_v3")):
        keys = sorted(set(D[f_]["drop"]) & set(D[v_]["drop"]))
        if not keys:
            continue
        for metric, field in (("success", "success"), ("recovery", "recovered")):
            x, y = paired_arrays(D[f_]["drop"], D[v_]["drop"], keys, field)
            r = row("drop", f_, x, keys, y, np.array([k[0] for k in keys]), metric)
            r["vs"] = v_
            r["v3_k"] = int(y.sum())
            fv.append(r)

    # PPS vs the task proxy alone, paired, on every test (faithful and v3)
    pvt = []
    for p_, t_ in (("pps_f_g04", "task_f"), ("pps_v3_g04", "task_v3")):
        for test, field_sets in (("normal", (("success", "success"),)), ("hard", (("success", "success"),)),
                                 ("drop", (("success", "success"), ("recovery", "recovered")))):
            a, b = D[p_][test], D[t_][test]
            keys = sorted(set(a) & set(b))
            if not keys:
                continue
            cl = (np.array(keys) % 50 if test == "normal" else np.array(keys) if test == "hard"
                  else np.array([k[0] for k in keys]))
            for metric, field in field_sets:
                x, y = paired_arrays(a, b, keys, field)
                r = row(test, p_, x, keys, y, cl, metric)
                r["vs"], r["vs_k"] = t_, int(y.sum())
                pvt.append(r)

    # trade-off plot on the common subset: normal trials 0-99, bank branch 0
    nkeys = list(range(100))
    dkeys = [(s, 0) for s in range(len(man["states"]))]

    def point(c):
        nn, dd = D[c]["normal"], D[c]["drop"]
        if not all(k in nn for k in nkeys) or not all(k in dd for k in dkeys):
            return None
        return (float(np.mean([nn[k]["success"] for k in nkeys])), float(np.mean([dd[k]["recovered"] for k in dkeys])),
                float(np.mean([dd[k]["success"] for k in dkeys])))
    pts = {c: point(c) for c in ["base"] + PPS + BLEND + ["task_f"]}
    plot_lines(root, pts)

    res = {"reuse_base_normal_from_nov11": reuse, "reuse_v3_normal_from_nov11": v3_reuse, "rows": rows, "retained_lost_acquired": rla_tab,
           "drop_by_source": by_src, "faithful_vs_v3": fv, "pps_vs_task": pvt, "tradeoff_points": pts,
           "cosine_check": cosine_check(root), "pad_audit": pad_audit(root, PPS + BLEND + ["task_f"]),
           "hold_steps": HOLD, "bank_sources": man["sources"]}
    (root / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    write_tables(root, res)
    print((root / "TABLES.md").read_text())


def plot_lines(root, pts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, muted, grid, surf = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
    blue, orange = "#2a78d6", "#eb6834"
    for ycol, fname, ylab in ((1, "tradeoff.png", "Drop-bank recovery (re-grasp held %d steps)" % HOLD),
                              (2, "tradeoff_drop_success.png", "Drop-bank task completion")):
        fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=150)
        fig.patch.set_facecolor(surf)
        ax.set_facecolor(surf)
        lines = [("PPS (faithful)", ["base"] + PPS, blue, "o", [0, .2, .4, .6, .8, 1.0]),
                 ("Blend (faithful)", ["base"] + BLEND + ["task_f"], orange, "s", [0, .2, .4, .6, .8, 1.0])]
        ys = [pts[c][ycol] for c in pts if pts.get(c)]
        for name, cs, col, mk, gs in lines:
            xy = [(pts[c][0], pts[c][ycol], g) for c, g in zip(cs, gs) if pts.get(c)]
            if len(xy) < 2:
                continue
            # the line starts at pi0.5 (shared by both lines, drawn separately below); markers only on its own points
            ax.plot([a for a, _, _ in xy], [b for _, b, _ in xy], color=col, lw=2, label=name, zorder=3, marker=mk,
                    ms=8, markeredgecolor=surf, markeredgewidth=2, markevery=list(range(1, len(xy))))
            is_pps = name.startswith("PPS")
            off = (-30, 8) if is_pps else (7, -13)  # PPS labels above, blend labels below
            for a, b, g in xy[1:-1]:
                ax.annotate("γ %.1f" % g, (a, b), xytext=off, textcoords="offset points", fontsize=8, color=ink2)
            a, b, g = xy[-1]
            if is_pps:
                end = "PPS γ %.1f" % g
            else:
                end = "task proxy alone\n(blend γ 1.0)" if g == 1.0 else "blend γ %.1f" % g
            ax.annotate(end, (a, b), xytext=(-62, 10) if is_pps else (8, -22), textcoords="offset points",
                        fontsize=9, color=ink)
        if pts.get("base"):
            ax.plot([pts["base"][0]], [pts["base"][ycol]], ls="none", marker="D", ms=8, color=ink2,
                    markeredgecolor=surf, markeredgewidth=2, zorder=5)
            ax.annotate("π0.5 (γ 0)", (pts["base"][0], pts["base"][ycol]), xytext=(-20, -20),
                        textcoords="offset points", fontsize=9, color=ink)
        ax.set_xlabel("Normal-trial task success (trials 0–99)", color=ink2)
        ax.set_ylabel(ylab + ", 120 states, branch 0", color=ink2)
        ax.set_xlim(-0.02, 1.08)
        ax.set_ylim(max(0.0, min(ys) - 0.1) if ys else 0.0, 1.02)
        ax.grid(True, color=grid, lw=0.6)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.tick_params(colors=muted)
        ax.legend(frameon=False, loc="lower left", labelcolor=ink2)
        ax.set_title("t59: task success vs " + ("recovery" if ycol == 1 else "drop completion") +
                     " across γ (RTX 4090)", color=ink, fontsize=11, loc="left")
        fig.tight_layout()
        fig.savefig(root / fname, facecolor=surf)
        plt.close(fig)


def write_tables(root, res):
    L = ["# Tables (generated by scripts/faithful/summarize_faithful.py)", ""]
    L += ["pi0.5 normal trials reused from the Nov 11 4090 run: %s" % res["reuse_base_normal_from_nov11"], ""]
    for test, metric, title in (("normal", "success", "Normal trials (paired starts)"),
                                ("drop", "success", "Drop bank, task completion"),
                                ("drop", "recovery", "Drop bank, recovery (re-grasp held %d steps)" % HOLD),
                                ("hard", "success", "Harder starts")):
        rs = [r for r in res["rows"] if r["test"] == test and r["metric"] == metric]
        if not rs:
            continue
        L += ["## " + title, "", "| Controller | Success, 95% Wilson | Δ vs π0.5, cluster bootstrap 95% | McNemar p (cell only / π0.5 only) |",
              "| --- | --- | --- | --- |"]
        for r in rs:
            s, d, m = fmt(r)
            L.append("| %s | %s | %s | %s |" % (r["name"], s, d, m))
        L.append("")
    L += ["## Drop bank by state source (rates)", "", "| Controller | " + " | ".join(
        "%s states: completion / recovery" % s["source"] for s in res["bank_sources"]) + " |",
          "| --- |" + " --- |" * len(res["bank_sources"])]
    for c, v in res["drop_by_source"].items():
        L.append("| %s | " % NAMES.get(c, c) + " | ".join(
            ("%.0f%% / %.0f%% (n %d)" % (100 * v[s["source"]]["success"], 100 * v[s["source"]]["recovery"], v[s["source"]]["n"])
             if v[s["source"]]["n"] else "—") for s in res["bank_sources"]) + " |")
    L.append("")
    for metric in ("success", "recovery"):
        L += ["## Retained / lost / acquired per (state, branch), %s" % metric, "",
              "| Comparison | Source | n | retained | lost | acquired | neither |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for k, v in res["retained_lost_acquired"].items():
            for src, t in v[metric].items():
                L.append("| %s | %s | %d | %d | %d | %d | %d |" % (k, src, t["n"], t["retained"], t["lost"], t["acquired"],
                                                                  t["neither"]))
        L.append("")
    if res["faithful_vs_v3"]:
        L += ["## Faithful vs v3 ours (paired on the same trials / bank branches)", "",
              "| Cell | Test | Metric | faithful | v3 ours | Δ faithful − v3, cluster 95% | McNemar p (faithful only / v3 only) |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
        for r in res["faithful_vs_v3"]:
            s, d, m = fmt(r)
            L.append("| %s | %s | %s | %s | %d/%d | %s | %s |" % (r["name"], r["test"], r["metric"], s, r["v3_k"], r["n"],
                                                                 d, m))
        L.append("")
    if res["pps_vs_task"]:
        L += ["## PPS γ 0.4 vs the task proxy alone (paired)", "",
              "| PPS cell | vs | Test | Metric | PPS | task alone | Δ PPS − task, cluster 95% | McNemar p (PPS only / task only) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in res["pps_vs_task"]:
            s, d, m = fmt(r)
            L.append("| %s | %s | %s | %s | %s | %d/%d | %s | %s |" % (r["name"], NAMES.get(r["vs"], r["vs"]), r["test"],
                                                                      r["metric"], s, r["vs_k"], r["n"], d, m))
        L.append("")
    L += ["## Trade-off points (normal 0–99, bank branch 0)", "", "| Cell | normal success | drop recovery | drop completion |",
          "| --- | --- | --- | --- |"]
    for c, v in res["tradeoff_points"].items():
        if v:
            L.append("| %s | %.2f | %.2f | %.2f |" % (NAMES.get(c, c), v[0], v[1], v[2]))
    L += ["", "Cosine check: " + json.dumps(res["cosine_check"]), "", "Padding audit: " + json.dumps(res["pad_audit"]), ""]
    (root / "TABLES.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
