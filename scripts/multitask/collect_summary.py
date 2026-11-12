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
    for t in (77, 82):
        if ("t%d_s0" % t) not in train:
            train["t%d_s0" % t] = {k: training(k, t, 0) for k in ("reference", "task")}
    train["t59_s0 (Nov 11, for reference)"] = {k: training(k, 59, 0) for k in ("reference", "task")}
    hard = {}
    for d in sorted(glob.glob(str(S / "hard" / "t*"))):
        t = int(pathlib.Path(d).name[1:])
        man = jload(pathlib.Path(d) / "manifest.json")
        if man is None:
            continue
        shutil.copy(pathlib.Path(d) / "manifest.json", OUT / "checks" / ("hard_t%d_manifest.json" % t))
        r = jload(pathlib.Path(d) / "result.json")
        if r is not None:
            shutil.copy(pathlib.Path(d) / "result.json", OUT / "checks" / ("hard_t%d_result.json" % t))
        hard[t] = {"result": r, "geometry": man["geometry"], "skipped": man["skipped"],
                   "tried": [{k: x[k] for k in ("direction", "unit_xy", "shift_m", "n_pass")} for x in man["tried"]]}
    pin = jload(S / "pin_test" / "pin_compare.json")
    if pin is not None:
        shutil.copy(S / "pin_test" / "pin_compare.json", OUT / "checks" / "memfix_pin_compare.json")
    logs = {}
    for name, f in (("chain", S / "stageB" / "logs" / "chain.log"), ("rest", S / "rest.log")):
        logs[name] = f.read_text().splitlines() if f.exists() else []
    for s in (1, 2):
        train["t59_s%d" % s] = {k: training(k, 59, s) for k in ("reference", "task")}
    out = {"gpu": GPU, "stageA": short, "stageB": stage_b, "stageB_not_evaluated": not_eval,
           "stageB_qualifying_so_far": chosen, "training": train, "hard_pilot": hard, "memfix_pin_compare": pin,
           "logs": logs, "step0_repro_t59": jload(OUT / "checks" / "step0_repro_t59.json")}
    (OUT / "screen_summary.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({"shortlist": (short or {}).get("shortlist"), "qualifying": chosen,
                      "stageB": {t: (r["base"]["successes"], r["task_f"]["successes"], r["qualifies"])
                                 for t, r in stage_b.items()}}, default=str))


if __name__ == "__main__":
    main()
