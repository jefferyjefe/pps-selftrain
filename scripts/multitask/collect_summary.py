"""Collect the stage 4 screen results into results/multitask/screen_summary.json and small check files. Plain Python 3.

Reads (never writes) /workspace/runs/stage4_screen/{stageA,stageB,step5} and the training run folders; copies the
small json checks into results/multitask/checks/. Run from the repo root.
"""

import glob
import json
import pathlib
import shutil

RUNS = pathlib.Path("/workspace/runs")
S = RUNS / "stage4_screen"
OUT = pathlib.Path("results/multitask")
GPU = "NVIDIA GeForce RTX 4090"


def jload(p):
    p = pathlib.Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def training(kind, task, seed):
    d = RUNS / ("%s_t%d_faithful_s%d" % (kind, task, seed))
    m, c = jload(d / "metrics.json"), jload(d / "config.json")
    if m is None or c is None or "train_s" not in m:
        return {"run": str(d), "complete": (d / "final.pt").exists(), "metrics": m is not None}
    return {"run": str(d), "complete": (d / "final.pt").exists(), "train_min": m["train_s"] / 60, "gpu": m["gpu"],
            "config_gpu": c["gpu"], "steps": c["steps"], "batch": c["batch"], "seed": c["seed"],
            "torch_seed": c["torch_seed"], "data": c.get("data"), "final_probe": m["probe"][-1] if m.get("probe") else None,
            "checks_pass": (c.get("checks") or {}).get("pass")}


def main():
    (OUT / "checks").mkdir(parents=True, exist_ok=True)
    short = jload(S / "stageA" / "shortlist.json")
    shutil.copy(S / "stageA" / "shortlist.json", OUT / "checks" / "stageA_shortlist.json")
    stage_b = {}
    for f in sorted(glob.glob(str(S / "stageB" / "t*" / "result.json"))):
        r = jload(f)
        t = r["task"]
        stage_b[t] = r
        for name in ("result.json", "repro_compare.json", "check_faithful.json"):
            src = S / "stageB" / ("t%d" % t) / name
            if src.exists():
                shutil.copy(src, OUT / "checks" / ("stageB_t%d_%s" % (t, name)))
    not_eval = {}
    for d in sorted(glob.glob(str(S / "stageB" / "t*"))):
        t = int(pathlib.Path(d).name[1:])
        if t not in stage_b:
            not_eval[t] = "CHECK_FAIL" if (pathlib.Path(d) / "CHECK_FAIL").exists() else "no result"
    chosen = [t for t in (short or {}).get("shortlist", []) if stage_b.get(t, {}).get("qualifies")][:3]
    train = {}
    for t in list(stage_b) + list(not_eval):
        train["t%d_s0" % t] = {k: training(k, t, 0) for k in ("reference", "task")}
    for s in (1, 2):
        train["t59_s%d" % s] = {k: training(k, 59, s) for k in ("reference", "task")}
    train["t59_s0 (Nov 11, for reference)"] = {k: training(k, 59, 0) for k in ("reference", "task")}
    cal = {}
    for f in sorted(glob.glob(str(S / "step5" / "t*" / "cal_result.json"))):
        r = jload(f)
        cal[r["task"]] = r
        shutil.copy(f, OUT / "checks" / ("step5_t%d_cal_result.json" % r["task"]))
        shutil.copy(S / "step5" / ("t%d" % r["task"]) / "pilot_states.json",
                    OUT / "checks" / ("step5_t%d_pilot_states.json" % r["task"]))
    out = {"gpu": GPU, "stageA": short, "stageB": stage_b, "stageB_not_evaluated": not_eval, "chosen": chosen,
           "training": train, "step5_calibration": cal,
           "step0_repro_t59": jload(OUT / "checks" / "step0_repro_t59.json")}
    (OUT / "screen_summary.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({"shortlist": (short or {}).get("shortlist"), "chosen": chosen,
                      "stageB": {t: (r["base"]["successes"], r["task_f"]["successes"], r["qualifies"])
                                 for t, r in stage_b.items()}}, default=str))


if __name__ == "__main__":
    main()
