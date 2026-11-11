"""Episode runner for the faithful-proxy t59 evaluation. openpi's examples/libero venv (Python 3.8).

Reuses scripts/stage3/run_t59.py and t59_sim.py unchanged (imported): the openpi LIBERO episode loop, trial setups
(trial i: init state i % 50, env.seed(7 + i), noise default_rng((7, i))), the mid-carry event, exact snapshots, the drop
(saved queue discarded, arm held with [0]*6 + [-1] and gripper forced open for 10 steps, no mj_forward on restore) and
the drop branch noise default_rng((11, s, b)). Differences from run_t59.py, all additive:
  - Cells: any server method and gamma (CELLS below), e.g. pps_f_g04 = faithful PPS at gamma 0.4.
  - The episode loop (rollout, copied from run_t59.py) also records the server's running padding-dim check counters
    for the faithful hook methods.
  - Normal trials of the bank source cells (base, pps_f_g04, task_f) snapshot their first mid-carry event to
    <out>/pool_src/<cell>/trial_XXXX.pkl, exactly as run_t59.py does for pi0.5.
  - Jobs come from <out>/queue.jsonl in file order (priority), claimed with O_EXCL files in <out>/claims, done-ness
    from each cell's jsonl. The queue can grow while workers run. A worker exits when no unclaimed job is left (it
    waits instead if <out>/MORE exists) or when <out>/STOP exists.
Job kinds: {"kind": "paired", "cell", "trial"}, {"kind": "drop", "cell", "state", "branch"} on the common bank
(<out>/bank/manifest.json), {"kind": "hard", "cell", "trial"} on the harder starts (<out>/hard/manifest.json).

Usage
  PYTHONPATH=.:/workspace/openpi/third_party/libero /workspace/openpi/examples/libero/.venv/bin/python \
      scripts/faithful/run_faithful.py worker --port 8020 --out /workspace/runs/eval_t59_faithful
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "stage3"))
import run_t59 as R  # noqa: E402
import t59_sim as S  # noqa: E402
from pps_real import libero_obs  # noqa: E402


def _cells():
    c = {"base": {"__method": "base"}, "task_f": {"__method": "task_f"},
         "pps_v3_g04": {"__method": "pps_v3", "__gamma": 0.4}, "task_v3": {"__method": "task_v3"}}
    for g in (0.2, 0.4, 0.6, 0.8, 1.0):
        c["pps_f_g%02d" % round(g * 10)] = {"__method": "pps_f", "__gamma": g}
    for g in (0.2, 0.4, 0.6, 0.8):
        c["blend_f_g%02d" % round(g * 10)] = {"__method": "blend_f", "__gamma": g}
    return c


CELLS = _cells()
R.METHODS.clear()
R.METHODS.update(CELLS)
POOL_CELLS = ("base", "pps_f_g04", "task_f")
VIDEO_CELLS = ("base", "pps_f_g04", "task_f", "blend_f_g04")


def rollout(env, client, prompt, probe, t, obs, plan, noise_rng, drop_frames=0, on_step=None, frames=None,
            method="base"):
    """run_t59.rollout, plus the server's padding check counters per call."""
    steps = {k: [] for k in R.STEP_KEYS}
    calls = {"t": [], "infer_ms": [], "pps": collections.defaultdict(list), "pad_checked": [], "pad_violations": []}
    sim = env.env.sim
    gripper = env.env.robots[0].gripper
    done, err = False, None
    drop_left = drop_frames
    if drop_frames:
        plan.clear()
    while t < S.T_END:
        try:
            img, wrist = libero_obs.process_images(obs)
            dropping = drop_left > 0
            if frames is not None:
                f = np.concatenate([img, wrist], axis=1)
                frames.append(R.border(f, (255, 0, 0)) if dropping else f)
            if dropping:
                action = list(S.DROP_ACTION)
                S.force_open(gripper)
                drop_left -= 1
            else:
                if not plan:
                    el = {"observation/image": img, "observation/wrist_image": wrist,
                          "observation/state": libero_obs.policy_state(obs), "prompt": prompt,
                          "__noise": noise_rng.standard_normal(libero_obs.NOISE_SHAPE).astype(np.float32)}
                    el.update(R.METHODS[method])
                    try:
                        out = client.infer(el)
                        assert out["server_stats"]["method"] == R.METHODS[method]["__method"]
                    except Exception as e:  # noqa: BLE001
                        raise R.ServerError(repr(e))
                    calls["t"].append(t)
                    calls["infer_ms"].append(out["server_stats"]["infer_ms"])
                    calls["pad_checked"].append(out["server_stats"].get("pad_checked", -1))
                    calls["pad_violations"].append(out["server_stats"].get("pad_violations", -1))
                    for k, v in (out.get("pps") or {}).items():
                        calls["pps"][k].append(np.asarray(v))
                    plan.extend(out["actions"][: S.REPLAN_STEPS])
                action = plan.popleft().tolist()
            obs, _, done, _ = env.step(action)
            r = probe.read(action)
            steps["t"].append(t)
            steps["action"].append(np.asarray(action, dtype=np.float64))
            steps["qpos"].append(sim.data.qpos.copy())
            steps["qvel"].append(sim.data.qvel.copy())
            for k in R.STEP_KEYS[4:]:
                steps[k].append(r[k])
            if done:
                break
            if on_step is not None:
                on_step(t, action, r, obs, plan, noise_rng, len(calls["t"]))
            t += 1
        except R.ServerError:
            raise
        except Exception as e:  # same policy as openpi main.py, but recorded
            err = repr(e)
            logging.exception("episode error")
            break
    return {"success": bool(done), "t_end": int(t), "error": err, "steps": steps, "calls": calls}


R.rollout = rollout  # run_t59.drop_branch looks it up at call time


R_save_npz_orig = R.save_npz


def _save_npz(path, res, extra=None):
    """run_t59.save_npz plus the per-call padding check counters (used for paired and, via R, drop branches)."""
    ex = {"call_pad_checked": np.asarray(res["calls"].get("pad_checked", [])),
          "call_pad_violations": np.asarray(res["calls"].get("pad_violations", []))}
    if extra:
        ex.update(extra)
    R_save_npz_orig(path, res, ex)


R.save_npz = _save_npz  # run_t59.drop_branch looks it up at call time


def _pad_summary(calls):
    v = [x for x in calls["pad_violations"] if x >= 0]
    c = [x for x in calls["pad_checked"] if x >= 0]
    return None if not v else {"server_pad_violations_last": int(v[-1]), "server_pad_checked_last": int(c[-1]),
                               "server_pad_violations_first": int(v[0])}


class Ctx(object):
    def __init__(self, args):
        self.env, self.task, self.init_states = S.make_env()
        self.prompt = str(self.task.language)
        self.client, self.meta = R.connect(args.port)
        self.port = args.port
        self.root = pathlib.Path(args.out)
        self.seed = args.seed
        self.pool = self.root / "bank"
        self.hard = None
        hm = self.root / "hard" / "manifest.json"
        if hm.exists():
            self.hard = json.loads(hm.read_text())


def paired_trial(ctx, cell, trial, kind="paired"):
    """run_t59.paired_trial for any cell; kind "hard" uses the harder start init states (hard/manifest.json)."""
    out = ctx.root / cell
    sub = "paired" if kind == "paired" else "hard"
    (out / sub).mkdir(parents=True, exist_ok=True)
    key = "trial_%04d" % trial
    w0 = time.monotonic()
    if kind == "paired":
        setup = S.trial_setup(trial, len(ctx.init_states))
        init_states = ctx.init_states
    else:
        h = ctx.hard["trials"][trial]
        setup = {"trial": int(trial), "init_state": int(trial), "env_seed": int(h["env_seed"]),
                 "noise_seed": [int(x) for x in h["noise_seed"]], "hard_base_init_state": int(h["base_init_state"])}
        init_states = np.load(str(ctx.root / "hard" / "init_states.npy"))
    obs, t = R.start_episode(ctx.env, init_states, setup)
    probe = S.Probe(ctx.env)
    probe.start()
    noise_rng = np.random.default_rng(tuple(setup["noise_seed"]))
    ev = {}
    pool = (ctx.root / "pool_src" / cell) if (kind == "paired" and cell in POOL_CELLS) else None
    if pool is not None:
        pool.mkdir(parents=True, exist_ok=True)

    def on_step(tt, action, r, obs_new, plan, rng, ncalls):
        if ev or not S.mid_carry(r):
            return
        ev.update({"step": int(tt), "probe": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()},
                   "plan_len": len(plan)})
        if pool is not None:
            img, wrist = libero_obs.process_images(obs_new)
            client_state = dict(setup, t_next=int(tt) + 1, plan=[np.array(a) for a in plan],
                                noise_rng_state=rng.bit_generator.state, n_calls=int(ncalls), z0=probe.z0,
                                event_step=int(tt), event_probe=ev["probe"], img=img, wrist=wrist,
                                source_method=cell, source_server=ctx.meta)
            tmp = pool / (key + ".pkl.tmp")
            S.save_snapshot(tmp, S.take_snapshot(ctx.env, client_state))
            os.replace(str(tmp), str(pool / (key + ".pkl")))

    video = kind == "paired" and trial in (0, 1) and cell in VIDEO_CELLS
    frames = [] if video else None
    res = rollout(ctx.env, ctx.client, ctx.prompt, probe, t, obs, collections.deque(), noise_rng, on_step=on_step,
                  frames=frames, method=cell)
    st = res["steps"]
    rec = dict(key=key, mode=kind, method=cell, **setup, success=res["success"], env_steps=res["t_end"],
               error=res["error"], policy_calls=len(res["calls"]["t"]),
               infer_ms_mean=float(np.mean(res["calls"]["infer_ms"][1:])) if len(res["calls"]["t"]) > 1 else None,
               wall_s=time.monotonic() - w0, z0=probe.z0, event=ev or None,
               first_grasp_step=R.first(st, lambda i: st["grasp"][i]),
               lift2cm_step=R.first(st, lambda i: st["dz"][i] > 0.02),
               max_dz=float(np.max(st["dz"])) if st["dz"] else None,
               pps=R.pps_summary(res["calls"]), pad=_pad_summary(res["calls"]), server=ctx.meta,
               gamma=CELLS[cell].get("__gamma"))
    _save_npz(out / sub / (key + ".npz"), res)
    R.append_jsonl(out / (sub + ".jsonl"), rec)
    if frames:
        import imageio
        vd = ctx.root / "videos"
        vd.mkdir(exist_ok=True)
        imageio.mimwrite(str(vd / ("%s_%s_%s_%s.mp4" % (cell, sub, key, "success" if res["success"] else "failure"))),
                         frames, fps=20)
    logging.info("%s %s %s success=%s steps=%d event=%s calls=%d wall=%.1fs", cell, sub, key, res["success"],
                 res["t_end"], ev.get("step"), rec["policy_calls"], rec["wall_s"])


def drop_branch(ctx, cell, man, s, b):
    """run_t59.drop_branch (unchanged, with this module's rollout and npz extras), plus the state's source."""
    video = (s, b) in ((0, 0), (40, 0), (80, 0)) and cell in VIDEO_CELLS
    R.drop_branch(ctx, cell, man, s, b, video=video)


def _job_key(j):
    if j["kind"] == "drop":
        return "%s/drop/state_%02d_branch_%d" % (j["cell"], j["state"], j["branch"])
    return "%s/%s/trial_%04d" % (j["cell"], j["kind"], j["trial"])


def _done_all(root):
    done = set()
    for jl, kind in (("paired.jsonl", "paired"), ("drop.jsonl", "drop"), ("hard.jsonl", "hard")):
        for c in CELLS:
            for k in R.done_keys(root / c / jl):
                done.add("%s/%s/%s" % (c, kind, k))
    return done


def worker(ctx, cells=None):
    """Take the first unclaimed, not yet done job in queue order whose cell this worker serves (cells) and whose
    inputs exist (bank state ids present, harder starts built). A server failure releases the claim and exits."""
    root = ctx.root
    man = None
    while True:
        if (root / "STOP").exists() or (root / ("STOP_%d" % ctx.port)).exists():
            return
        queue = [json.loads(l) for l in open(root / "queue.jsonl")] if (root / "queue.jsonl").exists() else []
        done = _done_all(root)
        mp = root / "bank" / "manifest.json"
        n_states = len(json.loads(mp.read_text())["states"]) if mp.exists() else 0
        job = None
        for j in queue:
            if cells is not None and j["cell"] not in cells:
                continue
            k = _job_key(j)
            if k in done:
                continue
            if j["kind"] == "drop" and j["state"] >= n_states:
                continue
            if j["kind"] == "hard" and ctx.hard is None:
                hm = root / "hard" / "manifest.json"
                if not hm.exists():
                    continue
                ctx.hard = json.loads(hm.read_text())
            if R._claim(root, k.replace("/", "__")):  # noqa: SLF001
                job = j
                break
        if job is None:
            if (root / "MORE").exists():
                time.sleep(15)
                continue
            return
        try:
            if job["kind"] == "drop":
                if man is None or len(man["states"]) <= job["state"]:
                    man = R.load_manifest(ctx.pool)
                drop_branch(ctx, job["cell"], man, job["state"], job["branch"])
            else:
                paired_trial(ctx, job["cell"], job["trial"], kind=job["kind"])
        except R.ServerError:
            os.unlink(str(root / "claims" / _job_key(job).replace("/", "__")))
            raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("worker",))
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=S.SEED)
    p.add_argument("--cells", default=None, help="comma separated cells this worker serves (default all)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ctx = Ctx(args)
    logging.info("t59 init states: %d, server %s", len(ctx.init_states), ctx.meta)
    worker(ctx, set(args.cells.split(",")) if args.cells else None)
    ctx.env.close()


if __name__ == "__main__":
    main()
