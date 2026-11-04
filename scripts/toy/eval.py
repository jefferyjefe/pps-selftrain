"""Step 6. Evaluate on the variant. Base alone, task proxy alone, reference proxy alone
(diagnostic), and PPS over a gamma sweep, for each proxy seed. Every condition uses the
same env seeds and the same sampler noise, so per-episode results pair up across
conditions. Writes results/toy_gamma_sweep.json and results/toy_gamma_sweep.png.

Primary metrics are mean max coverage and success at 0.90. Success at the env's own 0.95
threshold is kept in the JSON.
"""

import argparse

import toy  # noqa F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from toy.common import RESULTS_DIR, SEEDS, get_device, load_json, save_json
from toy.data import variant_offset
from toy.env import PRIMARY_THRESHOLD, fmt, rollout, summarize
from toy.pps import load_policies, make_policy

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C_PPS, C_BASE, C_TASK = "#2a78d6", "#eb6834", "#1baf7a"
METRICS = {
    "mean_max_coverage": ("mean max coverage", 1.0),
    "success_090": (f"success at {PRIMARY_THRESHOLD:.2f} (%)", 100.0),
}


def per_episode(entries: list[dict], metric: str) -> np.ndarray:
    """Per-episode values pooled over seeds, in a fixed (seed, episode) order."""
    mc = np.concatenate([np.array(e["max_coverage"]) for e in entries])
    return mc if metric == "mean_max_coverage" else (mc > PRIMARY_THRESHOLD).astype(float)


def aggregate(per_seed: list[dict]) -> dict:
    out = {"episodes_total": sum(s["n"] for s in per_seed)}
    for m in ("mean_max_coverage", "success_090", "success_095"):
        v = np.array([s[m] for s in per_seed])
        out[m] = float(v.mean())
        out[m + "_std_across_seeds"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        out[m + "_per_seed"] = v.tolist()
    return out


def paired(per_seed: dict, a: str, b: str, metric: str) -> dict:
    """Mean of a minus b over paired episodes, with its standard error."""
    d = per_episode(per_seed[a], metric) - per_episode(per_seed[b], metric)
    return {"diff": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d)))}


def peak_check(per_seed: dict, gammas: list[float]) -> dict:
    """A middle gamma peak means the best gamma lies strictly between 0 and 1 and beats both
    gamma 1 and the task proxy alone by more than two paired standard errors."""
    out = {}
    for m in METRICS:
        means = {g: float(per_episode(per_seed[f"pps_{g}"], m).mean()) for g in gammas}
        best = max(means, key=means.get)
        vs_one = paired(per_seed, f"pps_{best}", "pps_1.0", m) if best != 1.0 else None
        vs_task = paired(per_seed, f"pps_{best}", "task", m)
        interior = 0.0 < best < 1.0
        beats = lambda p: p is not None and p["diff"] > 2 * p["se"]
        out[m] = {"best_gamma": best, "best_value": means[best], "best_minus_gamma1": vs_one,
                  "best_minus_task": vs_task, "middle_peak": bool(interior and beats(vs_one) and beats(vs_task))}
    return out


def plot(out, gammas, offset, n_eps, seeds):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax, (m, (label, scale)) in zip(axes, METRICS.items()):
        ax.set_facecolor(SURFACE)
        pps = [out["pps"][str(g)] for g in gammas]
        y = scale * np.array([p[m] for p in pps])
        e = scale * np.array([p[m + "_std_across_seeds"] for p in pps])
        base = scale * out["base"][m]
        task = scale * out["task"][m]
        unit = "%" if scale == 100 else ""
        ax.axhline(base, color=C_BASE, lw=2, ls="--", label=f"base alone ({base:.{1 if unit else 3}f}{unit})")
        ax.axhline(task, color=C_TASK, lw=2, ls=":", label=f"task proxy alone ({task:.{1 if unit else 3}f}{unit})")
        ax.errorbar(gammas, y, yerr=e, color=C_PPS, lw=2, marker="o", ms=7, capsize=3,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label="PPS", zorder=3)
        for g, v, err in zip(gammas, y, e):
            # Label sits above the error bar cap so it never collides with the line or bar.
            ax.annotate(f"{v:.1f}" if unit else f"{v:.3f}", (g, v + err), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=7.5, color=INK2)
        ax.set_xlabel("steering strength gamma", color=INK)
        ax.set_ylabel(label, color=INK)
        ax.set_xticks(gammas)
        lo = min(y.min() - e.max(), base, task)
        hi = max(y.max() + e.max(), base, task)
        pad = 0.18 * (hi - lo)
        ax.set_ylim(max(0.0, lo - pad), hi + 1.6 * pad)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(INK2)
        ax.tick_params(colors=INK2)
        # Legend below the axes so it can never cover a line.
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    fig.suptitle(f"Toy PPS on PushT, goal shifted ({offset[0]:.0f}, {offset[1]:.0f}) px",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.text(0.01, 0.01, f"{n_eps} episodes per condition per seed, proxy seeds {seeds}. "
             "Points are means over seeds, error bars the std across seeds.", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    path = RESULTS_DIR / "toy_gamma_sweep.png"
    fig.savefig(path, facecolor=SURFACE)
    print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0], help="proxy training seeds")
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--plot-only", action="store_true", help="redraw the plot from the saved JSON")
    args = ap.parse_args()

    if args.plot_only:
        out = load_json(RESULTS_DIR / "toy_gamma_sweep.json")
        gammas = [float(g) for g in out["pps"]]
        plot(out, gammas, out["goal_offset"], out["episodes_per_condition_per_seed"], out["proxy_seeds"])
        return

    dev = get_device(args.device)
    offset = variant_offset()
    env_seeds = SEEDS["variant_eval"] + np.arange(args.episodes)

    def run(pol):
        res = rollout(pol, env_seeds, goal_offset=offset, device=dev, noise_seed=args.noise_seed)
        return {**summarize(res), "max_coverage": np.round(res["max_coverage"], 4).tolist()}

    per_seed = {"base": [], "task": [], "ref": [], **{f"pps_{g}": [] for g in args.gammas}}
    for seed in args.seeds:
        base, ref, task = load_policies(dev, seed)
        conds = [("base", make_policy("base", base=base)),
                 ("task", make_policy("task", task=task)),
                 ("ref", make_policy("ref", ref=ref))]
        conds += [(f"pps_{g}", make_policy("pps", base=base, task=task, ref=ref, gamma=g)) for g in args.gammas]
        for name, pol in conds:
            s = run(pol)
            per_seed[name].append({"seed": seed, **s})
            print(f"seed {seed} {name}: {fmt(s)}", flush=True)

    out = {
        "goal_offset": offset,
        "episodes_per_condition_per_seed": args.episodes,
        "env_seeds_start": int(env_seeds[0]),
        "noise_seed": args.noise_seed,
        "proxy_seeds": args.seeds,
        "primary_threshold": PRIMARY_THRESHOLD,
        "base": aggregate(per_seed["base"]),
        "task": aggregate(per_seed["task"]),
        "ref": aggregate(per_seed["ref"]),
        "pps": {str(g): aggregate(per_seed[f"pps_{g}"]) for g in args.gammas},
        "peak_check": peak_check(per_seed, args.gammas),
        "per_seed": per_seed,
    }
    save_json(out, RESULTS_DIR / "toy_gamma_sweep.json")
    plot(out, args.gammas, offset, args.episodes, args.seeds)
    for m in METRICS:
        row = [f"base {out['base'][m]:.3f}", f"task {out['task'][m]:.3f}", f"ref {out['ref'][m]:.3f}"]
        row += [f"pps({g}) {out['pps'][str(g)][m]:.3f}" for g in args.gammas]
        print(f"summary {m}: " + ", ".join(row))
        pc = out["peak_check"][m]
        print(f"peak check {m}: best gamma {pc['best_gamma']}, middle peak {pc['middle_peak']}, "
              f"best minus gamma 1 {pc['best_minus_gamma1']}, best minus task {pc['best_minus_task']}")


if __name__ == "__main__":
    main()
