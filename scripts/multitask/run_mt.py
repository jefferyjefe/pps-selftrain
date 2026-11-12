"""Episode worker for the stage 4 multi-task screen. openpi's examples/libero venv (Python 3.8).

The episode loop is scripts/faithful/run_faithful.py's rollout (imported, unchanged: 10 settle steps, t < 410, replan
every 5 of a 10 step chunk, success = env done), with sim_mt.Probe in place of the t59 probe, and the npz layout of
run_t59.save_npz plus run_faithful's padding counters. Cells are run_faithful.CELLS (base = pi0.5 alone, task_f = the
faithful task proxy alone, ...), served by scripts/faithful/serve_faithful.py.

Job kinds (from <out>/queue.jsonl in file order, claimed with O_EXCL files in <out>/claims, done-ness from each
(task, cell)'s jsonl):
  {"kind": "screen", "task": T, "cell": C, "trial": j}  screen trial j (sim_mt.trial_setup with SCREEN_OFFSET)
  {"kind": "pilot",  "task": T, "cell": C, "trial": j}  drop-calibration pilot trial j (PILOT_OFFSET); the first
                                                        mid-carry event is snapshotted to <out>/t<T>/pilot_src/<C>/
Output: <out>/t<T>/<C>/<kind>.jsonl and <out>/t<T>/<C>/<kind>/trial_XXXX.npz.

A worker serves the tasks in --tasks (default all) and the cells in --cells (default all). For any cell other than
base it requires the server's metadata task_id to equal the job's task (the server holds one task's proxies). A worker
keeps one env and rebuilds it when the job's task changes. It exits when no unclaimed job is left (it waits instead
while <out>/MORE exists) or when <out>/STOP or <out>/STOP_<port> exists. A server failure releases the claim and exits.

Usage
  PYTHONPATH=.:/workspace/openpi/third_party/libero /workspace/openpi/examples/libero/.venv/bin/python \
      scripts/multitask/run_mt.py --port 8020 --out /workspace/runs/stage4_screen/stageA
"""

import argparse
import collections
import json
import logging
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "faithful"))
sys.path.insert(0, str(HERE.parent / "stage3"))
import run_faithful as RF  # noqa: E402  (sets run_t59.METHODS to the faithful cells)
import run_t59 as R  # noqa: E402
import sim_mt as M  # noqa: E402
import t59_sim as S  # noqa: E402
from pps_real import libero_obs  # noqa: E402

OFFSETS = {"screen": M.SCREEN_OFFSET, "pilot": M.PILOT_OFFSET}


def mid_carry(r):
    """t59_sim.mid_carry's thresholds (LIFT_M, TRAY_MARGIN_M) on sim_mt.Probe.read records."""
    return S.mid_carry(r)


def grasp_onsets(st):
    """Env steps at which (both fingers on the object and gripper commanded closed) switches on."""
    out, prev = [], False
    for i, t in enumerate(st["t"]):
        g = bool(st["grasp"][i] and st["a6"][i] > 0)
        if g and not prev:
            out.append(int(t))
        prev = g
    return out


class Ctx(object):
    def __init__(self, args):
        self.client, self.meta = R.connect(args.port)
        self.port = args.port
        self.root = pathlib.Path(args.out)
        self.task_id = None
        self.env = None

    def use_task(self, task_id):
        if self.task_id == task_id:
            return
        if self.env is not None:
            self.env.close()
        self.env, self.task, self.init_states = M.make_env(task_id)
        self.spec = M.task_spec(task_id)
        self.prompt = str(self.task.language)
        assert self.prompt == self.spec["prompt"]
        self.task_id = task_id
        logging.info("task %d %s: %d init states, obj %s, target %s", task_id, self.spec["name"],
                     len(self.init_states), self.spec["obj"], self.spec["target"])


def run_trial(ctx, job):
    task, cell, trial, kind = int(job["task"]), job["cell"], int(job["trial"]), job["kind"]
    ctx.use_task(task)
    out = ctx.root / ("t%d" % task) / cell
    (out / kind).mkdir(parents=True, exist_ok=True)
    key = "trial_%04d" % trial
    w0 = time.monotonic()
    setup = M.trial_setup(trial, len(ctx.init_states), OFFSETS[kind])
    obs, t = R.start_episode(ctx.env, ctx.init_states, setup)
    probe = M.Probe(ctx.env, ctx.spec)
    probe.start()
    noise_rng = np.random.default_rng(tuple(setup["noise_seed"]))
    ev = {}
    pool = (ctx.root / ("t%d" % task) / "pilot_src" / cell) if kind == "pilot" else None
    if pool is not None:
        pool.mkdir(parents=True, exist_ok=True)

    def on_step(tt, action, r, obs_new, plan, rng, ncalls):
        if ev or not mid_carry(r):
            return
        ev.update({"step": int(tt), "probe": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()},
                   "plan_len": len(plan)})
        if pool is not None:
            img, wrist = libero_obs.process_images(obs_new)
            client_state = dict(setup, task=task, t_next=int(tt) + 1, plan=[np.array(a) for a in plan],
                                noise_rng_state=rng.bit_generator.state, n_calls=int(ncalls), z0=probe.z0,
                                event_step=int(tt), event_probe=ev["probe"], img=img, wrist=wrist,
                                source_method=cell, source_server=ctx.meta)
            tmp = pool / (key + ".pkl.tmp")
            S.save_snapshot(tmp, S.take_snapshot(ctx.env, client_state))
            os.replace(str(tmp), str(pool / (key + ".pkl")))

    res = RF.rollout(ctx.env, ctx.client, ctx.prompt, probe, t, obs, collections.deque(), noise_rng, on_step=on_step,
                     method=cell)
    st = res["steps"]
    rec = dict(key=key, mode=kind, task=task, task_name=ctx.spec["name"], scene=ctx.spec["scene"], method=cell,
               **setup, success=res["success"], env_steps=res["t_end"], error=res["error"],
               policy_calls=len(res["calls"]["t"]),
               infer_ms_mean=float(np.mean(res["calls"]["infer_ms"][1:])) if len(res["calls"]["t"]) > 1 else None,
               wall_s=time.monotonic() - w0, z0=probe.z0, obj=ctx.spec["obj"], target=ctx.spec["target"],
               event=ev or None, first_grasp_step=R.first(st, lambda i: st["grasp"][i]),
               grasp_onsets=grasp_onsets(st), lift2cm_step=R.first(st, lambda i: st["dz"][i] > 0.02),
               max_dz=float(np.max(st["dz"])) if st["dz"] else None,
               pps=R.pps_summary(res["calls"]), pad=RF._pad_summary(res["calls"]), server=ctx.meta,  # noqa: SLF001
               gamma=RF.CELLS[cell].get("__gamma"))
    RF._save_npz(out / kind / (key + ".npz"), res)  # noqa: SLF001
    R.append_jsonl(out / (kind + ".jsonl"), rec)
    logging.info("t%d %s %s %s success=%s steps=%d event=%s calls=%d wall=%.1fs", task, cell, kind, key,
                 res["success"], res["t_end"], ev.get("step"), rec["policy_calls"], rec["wall_s"])


def job_key(j):
    return "t%d/%s/%s/trial_%04d" % (int(j["task"]), j["cell"], j["kind"], int(j["trial"]))


def done_all(root, queue):
    done = set()
    for tc in sorted(set((int(j["task"]), j["cell"], j["kind"]) for j in queue)):
        for k in R.done_keys(root / ("t%d" % tc[0]) / tc[1] / (tc[2] + ".jsonl")):
            done.add("t%d/%s/%s/%s" % (tc[0], tc[1], tc[2], k))
    return done


def worker(ctx, tasks=None, cells=None):
    root = ctx.root
    server_task = ctx.meta.get("task_id")
    while True:
        if (root / "STOP").exists() or (root / ("STOP_%d" % ctx.port)).exists():
            return
        queue = [json.loads(l) for l in open(root / "queue.jsonl")] if (root / "queue.jsonl").exists() else []
        done = done_all(root, queue)
        job = None
        for j in queue:
            if tasks is not None and int(j["task"]) not in tasks:
                continue
            if cells is not None and j["cell"] not in cells:
                continue
            if j["cell"] != "base" and server_task != int(j["task"]):
                continue
            k = job_key(j)
            if k in done:
                continue
            if R._claim(root, k.replace("/", "__")):  # noqa: SLF001
                job = j
                break
        if job is None:
            if (root / "MORE").exists():
                time.sleep(15)
                continue
            return
        try:
            run_trial(ctx, job)
        except R.ServerError:
            os.unlink(str(root / "claims" / job_key(job).replace("/", "__")))
            raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tasks", default=None, help="comma separated task ids this worker serves (default all)")
    p.add_argument("--cells", default=None, help="comma separated cells this worker serves (default all)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ctx = Ctx(args)
    logging.info("server %s", ctx.meta)
    worker(ctx, set(int(x) for x in args.tasks.split(",")) if args.tasks else None,
           set(args.cells.split(",")) if args.cells else None)
    if ctx.env is not None:
        ctx.env.close()


if __name__ == "__main__":
    main()
