"""PPS line vs blend line for the faithful t59 sweep. openpi venv (CPU only).

Uses the same data, loader and statistics helpers as summarize_faithful.py, on the trade-off plot's trials: normal
trials 0-99 (pi0.5 reused from the Nov 11 4090 run) and drop-bank branch 0 of all 120 states.

1. Matched normal success. PPS and blend cells that reach the same normal-trial success (100/100) are compared paired on
   the 120 bank branches: difference in recovery and completion, cluster bootstrap 95% (clusters = bank state, one
   branch each) and exact McNemar p, plus retained / lost / acquired by state source.
2. Vertical gap. For each PPS point (x = normal success, y = recovery or completion), gap = y - B(x), where B(x) is the
   highest y the blend line (pi0.5 -> blend 0.2 ... 0.8 -> task alone, straight segments in gamma order) reaches at
   that x. 95% interval: joint bootstrap (10,000) that resamples normal-trial init states (50 clusters of 2 trials)
   and bank states (120) independently and recomputes every point and the blend line in each resample.
3. PPS sweep cells vs the task proxy alone on bank branch 0: retained / lost / acquired by state source.

Writes <root>/pps_vs_blend.json and <root>/pps_vs_blend.md.
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import summarize_faithful as S  # noqa: E402

ROOT = S.ROOT
RNG = np.random.default_rng(1)
NBOOT = 10000
PPS_LINE = ["base"] + S.PPS
BLEND_LINE = ["base"] + S.BLEND + ["task_f"]
MATCHED = [("pps_f_g06", "blend_f_g08"), ("pps_f_g08", "blend_f_g08"), ("pps_f_g06", "task_f"),
           ("pps_f_g08", "task_f"), ("pps_f_g06", "blend_f_g06"), ("pps_f_g04", "blend_f_g04")]
NAMES = dict(S.NAMES)
NAMES["base"] = "π0.5"


def blend_y_at(x, bx, by):
    """Highest y on the polyline (bx, by) (vertices in gamma order) at abscissa x; nan if x is outside its range."""
    best = np.nan
    for i in range(len(bx) - 1):
        x0, x1, y0, y1 = bx[i], bx[i + 1], by[i], by[i + 1]
        lo, hi = min(x0, x1), max(x0, x1)
        if not (lo - 1e-12 <= x <= hi + 1e-12):
            continue
        y = max(y0, y1) if abs(x1 - x0) < 1e-12 else y0 + (x - x0) * (y1 - y0) / (x1 - x0)
        best = y if np.isnan(best) else max(best, y)
    return best


def main():
    reuse_base = (ROOT / "repro_check" / "REUSE_OK").exists()
    cells = sorted(set(PPS_LINE + BLEND_LINE))
    D = {c: S.load_cell(ROOT, c, reuse_base) for c in cells}
    trials = list(range(100))
    states = list(range(120))
    for c in cells:
        assert all(t in D[c]["normal"] for t in trials), c
        assert all((s, 0) in D[c]["drop"] for s in states), c
    init = np.array([D["base"]["normal"][t]["init_state"] for t in trials])
    N = {c: np.array([float(D[c]["normal"][t]["success"]) for t in trials]) for c in cells}
    R = {c: np.array([float(D[c]["drop"][(s, 0)]["recovered"]) for s in states]) for c in cells}
    C = {c: np.array([float(D[c]["drop"][(s, 0)]["success"]) for s in states]) for c in cells}
    bank = json.load(open(ROOT / "bank" / "manifest.json")) if (ROOT / "bank" / "manifest.json").exists() else None
    src = np.array(["base"] * 40 + ["pps_f_g04"] * 40 + ["task_f"] * 40)

    out = {"n_normal": len(trials), "n_bank": len(states), "points": {}, "matched": [], "gaps": {},
           "pps_vs_task_branch0": {}}
    for c in cells:
        out["points"][c] = {"name": NAMES[c],
                            "normal": [int(N[c].sum()), len(trials), *S.wilson(N[c].sum(), len(trials))],
                            "recovery": [int(R[c].sum()), len(states), *S.wilson(R[c].sum(), len(states))],
                            "completion": [int(C[c].sum()), len(states), *S.wilson(C[c].sum(), len(states))]}

    def rla(a, b, Y):
        res = {}
        for name, m in (("all", np.ones(len(states), bool)), *((s, src == s) for s in ("base", "pps_f_g04", "task_f"))):
            x, y = Y[a][m], Y[b][m]
            res[name] = {"n": int(m.sum()), "retained": int(((x == 1) & (y == 1)).sum()),
                         "lost": int(((x == 0) & (y == 1)).sum()), "acquired": int(((x == 1) & (y == 0)).sum()),
                         "neither": int(((x == 0) & (y == 0)).sum())}
        return res

    cl = np.array(states)
    for a, b in MATCHED:
        row = {"a": a, "b": b, "normal": [int(N[a].sum()), int(N[b].sum()), len(trials)]}
        for metric, Y in (("recovery", R), ("completion", C)):
            d = float(Y[a].mean() - Y[b].mean())
            lo, hi = S.cluster_boot(Y[a], Y[b], cl, NBOOT)
            p, n10, n01 = S.mcnemar(Y[a], Y[b])
            row[metric] = {"a": int(Y[a].sum()), "b": int(Y[b].sum()), "n": len(states), "diff": d, "ci": [lo, hi],
                           "mcnemar_p": p, "a_only": n10, "b_only": n01, "rla": rla(a, b, Y)}
        out["matched"].append(row)

    # vertical gap of every PPS point to the blend line, joint bootstrap
    u_init = np.unique(init)
    init_members = [np.where(init == u)[0] for u in u_init]
    stacks = {}
    for metric, Y in (("recovery", R), ("completion", C)):
        def gaps(nidx, sidx):
            px = {c: N[c][nidx].mean() for c in cells}
            py = {c: Y[c][sidx].mean() for c in cells}
            bx = [px[c] for c in BLEND_LINE]
            by = [py[c] for c in BLEND_LINE]
            return {c: py[c] - blend_y_at(px[c], bx, by) for c in S.PPS}

        point = gaps(np.arange(len(trials)), np.arange(len(states)))
        bs = {c: [] for c in S.PPS}
        for _ in range(NBOOT):
            pick = RNG.integers(0, len(u_init), len(u_init))
            nidx = np.concatenate([init_members[i] for i in pick])
            sidx = RNG.integers(0, len(states), len(states))
            g = gaps(nidx, sidx)
            for c in S.PPS:
                bs[c].append(g[c])
        stacks[metric] = bs
        out["gaps"][metric] = {c: {"gap": float(point[c]), "ci": [float(np.nanpercentile(bs[c], 2.5)),
                                                                 float(np.nanpercentile(bs[c], 97.5))],
                                   "frac_boot_gt0": float(np.nanmean(np.array(bs[c]) > 0)),
                                   "n_boot_nan": int(np.isnan(bs[c]).sum())} for c in S.PPS}

    for c in S.PPS:
        out["pps_vs_task_branch0"][c] = {"recovery": rla(c, "task_f", R), "completion": rla(c, "task_f", C)}

    json.dump(out, open(ROOT / "pps_vs_blend.json", "w"), indent=1)

    f = lambda k, n, lo, hi: "%d/%d = %.1f%% [%.1f, %.1f]" % (k, n, 100 * k / n, 100 * lo, 100 * hi)  # noqa: E731
    L = ["# PPS line vs blend line (generated by scripts/faithful/pps_vs_blend.py)", "",
         "Normal trials 0–99 (n = 100 per point), bank branch 0 of 120 states (n = 120 per point).", "",
         "## Points with Wilson 95% intervals", "",
         "| Cell | normal success | drop recovery | drop completion |", "| --- | --- | --- | --- |"]
    for c in PPS_LINE + BLEND_LINE[1:]:
        p = out["points"][c]
        L.append("| %s | %s | %s | %s |" % (p["name"], f(*p["normal"]), f(*p["recovery"]), f(*p["completion"])))
    L += ["", "## Vertical gap: PPS point minus the best blend-line value at the same normal success", "",
          "| PPS cell | normal success | recovery gap, joint bootstrap 95% | share of resamples > 0 | completion gap, 95% | share > 0 |",
          "| --- | --- | --- | --- | --- | --- |"]
    for c in S.PPS:
        g, h = out["gaps"]["recovery"][c], out["gaps"]["completion"][c]
        L.append("| %s | %.2f | %+.1f [%+.1f, %+.1f] | %.3f | %+.1f [%+.1f, %+.1f] | %.3f |" % (
            NAMES[c], N[c].mean(), 100 * g["gap"], 100 * g["ci"][0], 100 * g["ci"][1], g["frac_boot_gt0"],
            100 * h["gap"], 100 * h["ci"][0], 100 * h["ci"][1], h["frac_boot_gt0"]))
    L += ["", "## Paired on the 120 bank branches", "",
          "| PPS cell | vs | normal success (both, n 100) | metric | PPS | other | Δ, cluster bootstrap 95% | McNemar p (PPS only / other only) |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in out["matched"]:
        for metric in ("recovery", "completion"):
            m = row[metric]
            L.append("| %s | %s | %d / %d | %s | %d/120 | %d/120 | %+.1f [%+.1f, %+.1f] | %.2g (%d / %d) |" % (
                NAMES[row["a"]], NAMES[row["b"]], row["normal"][0], row["normal"][1], metric, m["a"], m["b"],
                100 * m["diff"], 100 * m["ci"][0], 100 * m["ci"][1], m["mcnemar_p"], m["a_only"], m["b_only"]))
    L += ["", "## Retained / lost / acquired by state source, branch 0 (PPS vs other; lost = other only, acquired = PPS only)", "",
          "| Comparison | metric | source | n | retained | lost | acquired | neither |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    rows = [(r["a"], r["b"], r[m]["rla"], m) for r in out["matched"] for m in ("recovery", "completion")]
    rows += [(c, "task_f", out["pps_vs_task_branch0"][c][m], m) for c in S.PPS for m in ("recovery", "completion")]
    for a, b, res, m in rows:
        for s, v in res.items():
            L.append("| %s vs %s | %s | %s | %d | %d | %d | %d | %d |" % (
                NAMES[a], NAMES[b], m, s, v["n"], v["retained"], v["lost"], v["acquired"], v["neither"]))
    (ROOT / "pps_vs_blend.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
