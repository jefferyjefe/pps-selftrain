"""Loss curves of the all-50 paper style reference and task proxy runs (two panels, the losses differ in definition).

Reference: training loss (Eq. 5, per step MSE summed over the 10 steps) every 100 steps from train.log, and the
fixed train probe (505 frames, all draws and steps, same loss) every 1000 steps from metrics.json.
Task: training loss (openpi flow matching MSE mean over (10, 32)) every 100 steps, and the fixed probe (512 frames,
fixed noise and t) every 1000 steps, from metrics.json.

Usage
  /workspace/openpi/.venv/bin/python scripts/stage2/plot_paper_all50_curves.py
"""

import argparse
import json
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ("#2a78d6", "#eb6834")  # categorical slots 1, 2
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def ref_curves(d):
    tr = [(int(m[1]), float(m[2])) for m in re.finditer(r"step (\d+) loss\(sum over 10 steps\) ([\d.]+)",
                                                         (d / "train.log").read_text())]
    hist = json.loads((d / "metrics.json").read_text())["history"]
    pr = [(h["step"], h["train_probe"]["loss_sum_over_steps"]) for h in hist]
    return tr, pr


def task_curves(d):
    m = json.loads((d / "metrics.json").read_text())
    return [(r["step"], r["loss"]) for r in m["train"]], [(r["step"], r["probe_loss"]) for r in m["probe"]]


def panel(ax, title, ylabel, tr, pr, probe_label):
    ax.set_facecolor(SURFACE)
    ax.plot(*zip(*tr), color=SERIES[0], lw=2, label="training loss (mean of last 100 steps)")
    ax.plot(*zip(*pr), "o", ms=8, color=SERIES[1], markeredgecolor=SURFACE, markeredgewidth=2, label=probe_label)
    ax.set_yscale("log")
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel("step", color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.tick_params(colors=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9)
    s, v = tr[-1]
    ax.annotate(f"{v:.4f}", (s, v), xytext=(-4, 8), textcoords="offset points", ha="right", color=INK, fontsize=9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", default="/workspace/runs/reference_t59_paper_all50")
    p.add_argument("--task", default="/workspace/runs/task_t59_paper_all50")
    p.add_argument("--out", default="/workspace/runs/task_t59_paper_all50/loss_curves.png")
    args = p.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=SURFACE)
    tr, pr = ref_curves(pathlib.Path(args.ref))
    panel(axes[0], "Reference proxy, all 50 demos (Eq. 5 distillation)", "loss, per step MSE summed over 10 steps",
          tr, pr, "fixed train probe (505 frames)")
    tr2, pr2 = task_curves(pathlib.Path(args.task))
    panel(axes[1], "Task proxy, all 50 demos (openpi flow matching)", "loss, MSE mean over (10, 32)",
          tr2, pr2, "fixed probe (512 frames, fixed noise and t)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, facecolor=SURFACE)
    print(args.out)


if __name__ == "__main__":
    main()
