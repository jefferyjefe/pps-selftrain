"""Stage 3 t59 episode runner. Runs in openpi's examples/libero venv (Python 3.8). See t59_sim.py for conventions.

Modes
  paired   Trials --trials A B (B exclusive) or --trial-list. Trial i: init state i % 50, env.seed(seed + i), the j-th
           policy call gets the j-th (10, 32) draw of default_rng((seed, i)). With --pool-dir, the first mid-carry
           event of each trial is snapshotted (no change to the episode) to pool-dir/trial_XXXX.pkl.
  drop     Branches of pooled snapshots. --states (pool manifest state ids) x --branches. Branch b of state s restores
           the snapshot, drops the bottle (gripper's finger target forced fully open and action [0]*6 + [-1] for
           DROP_FRAMES env steps, the saved action queue is discarded), then the policy replans from the observation and
           runs until success or the trial's original budget (t < 410) ends. Policy noise: default_rng((DROP_SEED, s, b)).
  replay   Exactness check. Restore each snapshot with its saved action queue and noise generator state, continue the
           policy, and compare every step (action, qpos, qvel, bottle pose) with the original trial's log, bit for bit.
  repeat   Drop branch determinism check: run given (state, branch) pairs and write them, to compare with a second run.

Output (appended after every trial, resumable): <out>/<name>.jsonl and per trial <out>/<name>/<key>.npz.

Usage
  PYTHONPATH=.:/workspace/openpi/third_party/libero /workspace/openpi/examples/libero/.venv/bin/python \
      scripts/stage3/run_t59.py paired --port 8010 --method base --trials 0 200 --out /workspace/runs/eval_t59/base
"""

import argparse
import collections
import fcntl
import json
import logging
import os
import pathlib
import sys
import time

import imageio
import numpy as np
from openpi_client import websocket_client_policy as _wcp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t59_sim as S  # noqa: E402
from pps_real import libero_obs  # noqa: E402

DROP_SEED = 11
METHODS = {"base": {"__method": "base"}, "pps": {"__method": "pps", "__gamma": 0.4},
           "task": {"__method": "task"}}  # task: the task proxy alone (serve_eval.py), added for the 4090 rerun
ROOT = pathlib.Path("/workspace/runs/eval_t59")
N_PAIRED = 200
N_STATES, N_BRANCHES = 40, 5
VIDEO_TRIALS = (0, 1)
VIDEO_DROPS = ((0, 0), (1, 0))


class ServerError(RuntimeError):
    """Policy server unreachable or failed. Not an episode outcome: the worker exits and the job is retried."""
STEP_KEYS = ("t", "action", "qpos", "qvel", "obj", "dz", "grasp", "a6", "w", "tw", "table", "tray", "finger", "tray_dxy")


def border(img, color):
    img = img.copy()
    img[:4], img[-4:], img[:, :4], img[:, -4:] = color, color, color, color
    return img


def done_keys(path):
    out = set()
    if os.path.exists(path):
        for line in open(path):
            out.add(json.loads(line)["key"])
    return out


def append_jsonl(path, rec):
    with open(path, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
        fcntl.flock(fh, fcntl.LOCK_UN)


def rollout(env, client, prompt, probe, t, obs, plan, noise_rng, drop_frames=0, on_step=None, frames=None,
            method="base"):
    """openpi's LIBERO loop from env step t (past the settle steps). Returns the step and call logs."""
    steps = {k: [] for k in STEP_KEYS}
    calls = {"t": [], "infer_ms": [], "pps": collections.defaultdict(list)}
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
                frames.append(border(f, (255, 0, 0)) if dropping else f)
            if dropping:
                action = list(S.DROP_ACTION)
                S.force_open(gripper)
                drop_left -= 1
            else:
                if not plan:
                    el = {"observation/image": img, "observation/wrist_image": wrist,
                          "observation/state": libero_obs.policy_state(obs), "prompt": prompt,
                          "__noise": noise_rng.standard_normal(libero_obs.NOISE_SHAPE).astype(np.float32)}
                    el.update(METHODS[method])
                    try:
                        out = client.infer(el)
                        assert out["server_stats"]["method"] == METHODS[method]["__method"]
                    except Exception as e:  # noqa: BLE001
                        raise ServerError(repr(e))
                    calls["t"].append(t)
                    calls["infer_ms"].append(out["server_stats"]["infer_ms"])
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
            for k in STEP_KEYS[4:]:
                steps[k].append(r[k])
            if done:
                break
            if on_step is not None:
                on_step(t, action, r, obs, plan, noise_rng, len(calls["t"]))
            t += 1
        except ServerError:
            raise
        except Exception as e:  # same policy as openpi main.py, but recorded
            err = repr(e)
            logging.exception("episode error")
            break
    return {"success": bool(done), "t_end": int(t), "error": err, "steps": steps, "calls": calls}


def save_npz(path, res, extra=None):
    st = {("step_" + k): np.asarray(v) for k, v in res["steps"].items()}
    cl = {"call_t": np.asarray(res["calls"]["t"]), "call_infer_ms": np.asarray(res["calls"]["infer_ms"])}
    for k, v in res["calls"]["pps"].items():
        cl["pps_" + k] = np.stack(v)
    st.update(cl)
    if extra:
        st.update(extra)
    tmp = str(path) + ".tmp.npz"
    np.savez_compressed(tmp, **st)
    os.replace(tmp, path)


def pps_summary(calls):
    if not calls["pps"]:
        return None
    return {k: float(np.mean(np.stack(v))) for k, v in calls["pps"].items()}


def first(steps, pred, after=None):
    for i, tt in enumerate(steps["t"]):
        if (after is None or tt > after) and pred(i):
            return int(tt)
    return None


def connect(port):
    client = _wcp.WebsocketClientPolicy("127.0.0.1", port)
    return client, client.get_server_metadata().get("eval", {})


def start_episode(env, init_states, setup):
    env.seed(setup["env_seed"])
    env.reset()
    obs = env.set_init_state(init_states[setup["init_state"]])
    t = 0
    while t < S.SETTLE:
        obs, _, _, _ = env.step(libero_obs.DUMMY_ACTION)
        t += 1
    return obs, t


class Ctx(object):
    def __init__(self, args):
        self.env, self.task, self.init_states = S.make_env()
        self.prompt = str(self.task.language)
        self.client, self.meta = connect(args.port)
        self.root = pathlib.Path(args.out)
        self.seed = args.seed
        # Defaults reproduce the Nov 10 layout (one pool dir under --out, originals under --out).
        self.pool = pathlib.Path(args.pool_dir) if args.pool_dir else self.root / "pool"
        self.save_pool_dir = pathlib.Path(args.save_pool_dir) if args.save_pool_dir else self.pool
        self.orig_root = pathlib.Path(args.orig_root) if args.orig_root else self.root


def paired_trial(ctx, method, trial, video=False, save_pool=True):
    out = ctx.root / method
    (out / "paired").mkdir(parents=True, exist_ok=True)
    key = "trial_%04d" % trial
    w0 = time.monotonic()
    setup = S.trial_setup(trial, len(ctx.init_states))
    setup["noise_seed"] = [ctx.seed, trial]
    setup["env_seed"] = ctx.seed + trial
    obs, t = start_episode(ctx.env, ctx.init_states, setup)
    probe = S.Probe(ctx.env)
    probe.start()
    noise_rng = np.random.default_rng((ctx.seed, trial))
    ev = {}
    pool = ctx.save_pool_dir if (save_pool and method == "base") else None
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
                                source_method=method, source_server=ctx.meta)
            tmp = pool / (key + ".pkl.tmp")
            S.save_snapshot(tmp, S.take_snapshot(ctx.env, client_state))
            os.replace(str(tmp), str(pool / (key + ".pkl")))

    frames = [] if video else None
    res = rollout(ctx.env, ctx.client, ctx.prompt, probe, t, obs, collections.deque(), noise_rng, on_step=on_step,
                  frames=frames, method=method)
    st = res["steps"]
    rec = dict(key=key, mode="paired", method=method, **setup, success=res["success"], env_steps=res["t_end"],
               error=res["error"], policy_calls=len(res["calls"]["t"]),
               infer_ms_mean=float(np.mean(res["calls"]["infer_ms"][1:])) if len(res["calls"]["t"]) > 1 else None,
               wall_s=time.monotonic() - w0, z0=probe.z0, event=ev or None,
               first_grasp_step=first(st, lambda i: st["grasp"][i]),
               lift2cm_step=first(st, lambda i: st["dz"][i] > 0.02),
               max_dz=float(np.max(st["dz"])) if st["dz"] else None,
               pps=pps_summary(res["calls"]), server=ctx.meta, gamma=METHODS[method].get("__gamma"))
    save_npz(out / "paired" / (key + ".npz"), res)
    append_jsonl(out / "paired.jsonl", rec)
    if frames:
        vd = ctx.root / "videos"
        vd.mkdir(exist_ok=True)
        imageio.mimwrite(str(vd / ("%s_paired_%s_%s.mp4" % (method, key, "success" if res["success"] else "failure"))),
                         frames, fps=20)
    logging.info("%s %s success=%s steps=%d event=%s calls=%d wall=%.1fs", method, key, res["success"],
                 res["t_end"], ev.get("step"), rec["policy_calls"], rec["wall_s"])


def load_manifest(pool_dir):
    return json.loads((pathlib.Path(pool_dir) / "manifest.json").read_text())


def restore(env, init_states, snap):
    obs = S.restore_snapshot(env, init_states, snap)
    probe = S.Probe(env)
    probe.z0 = snap["client"]["z0"]
    probe.pert.z0 = snap["client"]["z0"]
    return obs, probe


def drop_branch(ctx, method, man, s, b, video=False, jl_name="drop.jsonl", tag=""):
    out = ctx.root / method
    (out / "drop").mkdir(parents=True, exist_ok=True)
    key = "state_%02d_branch_%d" % (s, b)
    w0 = time.monotonic()
    entry = man["states"][s]
    snap = S.load_snapshot(ctx.pool / entry["file"])
    c = snap["client"]
    obs, probe = restore(ctx.env, ctx.init_states, snap)
    noise_rng = np.random.default_rng((DROP_SEED, s, b))
    frames = [] if video else None
    res = rollout(ctx.env, ctx.client, ctx.prompt, probe, c["t_next"], obs, collections.deque(), noise_rng,
                  drop_frames=S.DROP_FRAMES, frames=frames, method=method)
    st = res["steps"]
    n = len(st["t"])
    hand = S.DROP_FRAMES - 1  # index of the last drop frame in the step log
    handover = {k: (st[k][hand].tolist() if isinstance(st[k][hand], np.ndarray) else st[k][hand])
                for k in STEP_KEYS[4:]} if n > hand else None
    after = st["t"][hand] if n > hand else None
    rec = dict(key=key, mode="drop", tag=tag, method=method, state=s, branch=b, source_trial=entry["trial"],
               noise_seed=[DROP_SEED, s, b], t_start=c["t_next"], budget_steps=S.T_END - c["t_next"],
               drop_frames=S.DROP_FRAMES, success=res["success"], env_steps=res["t_end"],
               steps_after_restore=n, error=res["error"], policy_calls=len(res["calls"]["t"]),
               handover=handover,
               bottle_touches_table_step=first(st, lambda i: st["table"][i]),
               regrasp_step=first(st, lambda i: st["grasp"][i] and st["a6"][i] > 0, after=after),
               relift_step=first(st, lambda i: st["dz"][i] >= S.LIFT_M and not st["table"][i], after=after),
               success_during_drop=bool(res["success"] and n <= S.DROP_FRAMES),
               wall_s=time.monotonic() - w0, pps=pps_summary(res["calls"]), server=ctx.meta,
               gamma=METHODS[method].get("__gamma"))
    save_npz(out / "drop" / (key + tag + ".npz"), res)
    append_jsonl(out / jl_name, rec)
    if frames:
        vd = ctx.root / "videos"
        vd.mkdir(exist_ok=True)
        imageio.mimwrite(str(vd / ("%s_drop_%s_%s.mp4" % (method, key, "success" if res["success"] else "failure"))),
                         frames, fps=20)
    logging.info("%s drop %s success=%s steps=%d regrasp=%s wall=%.1fs", method, key, res["success"], res["t_end"],
                 rec["regrasp_step"], rec["wall_s"])


def replay_state(ctx, man, s, jl):
    """Restore with the saved queue and noise generator state, continue pi0.5, compare with the original trial."""
    entry = man["states"][s]
    snap = S.load_snapshot(ctx.pool / entry["file"])
    c = snap["client"]
    orig = np.load(ctx.orig_root / "base" / "paired" / ("trial_%04d.npz" % entry["trial"]))
    w0 = time.monotonic()
    obs, probe = restore(ctx.env, ctx.init_states, snap)
    img, wrist = libero_obs.process_images(obs)
    img_equal = bool(np.array_equal(img, c["img"]) and np.array_equal(wrist, c["wrist"]))
    rng = np.random.default_rng()
    rng.bit_generator.state = c["noise_rng_state"]
    res = rollout(ctx.env, ctx.client, ctx.prompt, probe, c["t_next"], obs, collections.deque(c["plan"]), rng,
                  method="base")
    st = res["steps"]
    ot = orig["step_t"]
    hit = np.nonzero(ot == c["t_next"])[0]
    i0 = int(hit[0]) if len(hit) else None
    cmp = {}
    ok = img_equal and i0 is not None
    if i0 is not None:
        cmp["n_steps_orig"], cmp["n_steps_replay"] = int(len(ot) - i0), len(st["t"])
        ok = ok and cmp["n_steps_orig"] == cmp["n_steps_replay"]
        for k in ("action", "qpos", "qvel", "obj"):
            a, b = orig["step_" + k][i0:], np.asarray(st[k])
            same = a.shape == b.shape and np.array_equal(a, b)
            cmp[k + "_identical"] = bool(same)
            if not same and a.shape[1:] == b.shape[1:]:
                m = min(len(a), len(b))
                diff = np.abs(a[:m] - b[:m]).reshape(m, -1).max(1)
                nz = np.nonzero(diff > 0)[0]
                cmp[k + "_first_diff_step"] = int(st["t"][nz[0]]) if len(nz) else None
                cmp[k + "_max_abs_diff"] = float(diff.max()) if m else None
            ok = ok and same
    orig_rec = None
    for line in open(ctx.orig_root / "base" / "paired.jsonl"):
        r = json.loads(line)
        if r["trial"] == entry["trial"]:
            orig_rec = r
    same_outcome = orig_rec is not None and orig_rec["success"] == res["success"] and orig_rec["env_steps"] == res["t_end"]
    first_call = int(c["t_next"]) + len(c["plan"])  # env step of the replay's first fresh policy call
    diffs = [cmp.get(k + "_first_diff_step") for k in ("action", "qpos", "qvel", "obj") if not cmp.get(k + "_identical")]
    diffs = [d for d in diffs if d is not None]
    cmp["first_fresh_call_step"] = first_call
    cmp["first_diff_step_any"] = min(diffs) if diffs else None
    cmp["identical_before_first_fresh_call"] = bool(img_equal and i0 is not None and (not diffs or min(diffs) >= first_call))
    rec = dict(key="state_%02d" % s, state=s, trial=entry["trial"], t_next=c["t_next"], plan_len=len(c["plan"]),
               restored_images_identical=img_equal, identical=bool(ok and same_outcome), same_outcome=same_outcome,
               success=res["success"], env_steps=res["t_end"], orig_success=orig_rec and orig_rec["success"],
               orig_env_steps=orig_rec and orig_rec["env_steps"], orig_server_port=orig_rec and orig_rec["server"].get("port"),
               replay_server_port=ctx.meta.get("port"), wall_s=time.monotonic() - w0, **cmp)
    append_jsonl(jl, rec)
    logging.info("replay state %d trial %d identical=%s %s", s, entry["trial"], rec["identical"], cmp)
    return rec["identical"]


def simreplay_state(ctx, man, s, jl):
    """Sim side exactness, independent of the policy and the GPU. Restore the snapshot and step the env open loop with
    the original trial's logged actions from the snapshot on; compare qpos, qvel and bottle pose at every step and the
    outcome (done at the same step) with the original log, bit for bit. Added for the 4090 rerun: a policy in the loop
    replay cannot match a trajectory made on another GPU (bf16 kernels differ), but the sim and the restore must."""
    entry = man["states"][s]
    snap = S.load_snapshot(ctx.pool / entry["file"])
    c = snap["client"]
    orig = np.load(ctx.orig_root / "base" / "paired" / ("trial_%04d.npz" % entry["trial"]))
    w0 = time.monotonic()
    obs, probe = restore(ctx.env, ctx.init_states, snap)
    img, wrist = libero_obs.process_images(obs)
    img_equal = bool(np.array_equal(img, c["img"]) and np.array_equal(wrist, c["wrist"]))
    ot = orig["step_t"]
    i0 = int(np.nonzero(ot == c["t_next"])[0][0])
    acts = orig["step_action"][i0:]
    st = {"qpos": [], "qvel": [], "obj": []}
    done, t = False, int(c["t_next"])
    for a in acts:
        libero_obs.process_images(obs)  # as the rollout loop does (no effect on the sim)
        obs, _, done, _ = ctx.env.step(a.tolist())
        r = probe.read(a.tolist())
        st["qpos"].append(ctx.env.env.sim.data.qpos.copy())
        st["qvel"].append(ctx.env.env.sim.data.qvel.copy())
        st["obj"].append(r["obj"])
        if done:
            break
        t += 1
    cmp = {"n_steps_orig": int(len(acts)), "n_steps_replay": len(st["qpos"])}
    ok = img_equal and cmp["n_steps_orig"] == cmp["n_steps_replay"]
    for k in ("qpos", "qvel", "obj"):
        a, b = orig["step_" + k][i0:], np.asarray(st[k])
        same = a.shape == b.shape and np.array_equal(a, b)
        cmp[k + "_identical"] = bool(same)
        if not same and a.shape[1:] == b.shape[1:]:
            m = min(len(a), len(b))
            diff = np.abs(a[:m] - b[:m]).reshape(m, -1).max(1)
            nz = np.nonzero(diff > 0)[0]
            cmp[k + "_first_diff_index"] = int(nz[0]) if len(nz) else None
            cmp[k + "_max_abs_diff"] = float(diff.max()) if m else None
        ok = ok and same
    orig_rec = None
    for line in open(ctx.orig_root / "base" / "paired.jsonl"):
        r = json.loads(line)
        if r["trial"] == entry["trial"]:
            orig_rec = r
    t_end = t if done else S.T_END
    same_outcome = orig_rec is not None and orig_rec["success"] == bool(done) and orig_rec["env_steps"] == t_end
    rec = dict(key="state_%02d" % s, state=s, trial=entry["trial"], t_next=c["t_next"],
               restored_images_identical=img_equal, identical=bool(ok and same_outcome), same_outcome=same_outcome,
               success=bool(done), env_steps=t_end, orig_success=orig_rec and orig_rec["success"],
               orig_env_steps=orig_rec and orig_rec["env_steps"], wall_s=time.monotonic() - w0, **cmp)
    append_jsonl(jl, rec)
    logging.info("simreplay state %d trial %d identical=%s %s", s, entry["trial"], rec["identical"], cmp)
    return rec["identical"]


# ---------------------------------------------------------------- worker

def _done(root, method, jl):
    return done_keys(root / method / jl)


def _claim(root, name):
    d = root / "claims"
    d.mkdir(exist_ok=True)
    try:
        fd = os.open(str(d / name), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError:
        return False
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return True


def paired_jobs():
    return [(m, t) for t in range(N_PAIRED) for m in ("base", "pps")]


def drop_jobs(n_states=N_STATES):
    return [(m, s, b) for b in range(N_BRANCHES) for s in range(n_states) for m in ("base", "pps")]


def worker(ctx):
    """Pull jobs until none are left. Paired trials first (alternating methods), then drop branches once the
    orchestrator has written DROP_OK (pool selected and branching verified)."""
    root = ctx.root
    while True:
        did = False
        done = {m: _done(root, m, "paired.jsonl") for m in METHODS}
        for m, t in paired_jobs():
            key = "trial_%04d" % t
            if key in done[m] or not _claim(root, "paired_%s_%s" % (m, key)):
                continue
            paired_trial(ctx, m, t, video=t in VIDEO_TRIALS)
            did = True
            break
        if did:
            continue
        if (root / "POOL_EXTRA").exists() and not (root / "DROP_OK").exists():
            # pi0.5 trials 200..259, run only to fill the drop pool (kept out of the normal trial statistics)
            for t in range(N_PAIRED, N_PAIRED + 60):
                key = "trial_%04d" % t
                if key in done["base"] or not _claim(root, "paired_base_%s" % key):
                    continue
                paired_trial(ctx, "base", t)
                did = True
                break
            if did or not (root / "POOL_EXTRA").exists():
                continue
        if (root / "DROP_FAIL").exists() or (root / "STOP").exists():
            return
        if not (root / "DROP_OK").exists():
            time.sleep(10)
            continue
        man = load_manifest(ctx.pool)
        done = {m: _done(root, m, "drop.jsonl") for m in METHODS}
        for m, s, b in drop_jobs(len(man["states"])):
            key = "state_%02d_branch_%d" % (s, b)
            if key in done[m] or not _claim(root, "drop_%s_%s" % (m, key)):
                continue
            drop_branch(ctx, m, man, s, b, video=(s, b) in VIDEO_DROPS)
            did = True
            break
        if not did:
            return


def worker_groups(ctx, groups):
    """Worker for the 4090 rerun (pool fixed in advance, so no pool selection or POOL_EXTRA). Method groups run in
    order: the first group's paired trials, then its drop branches (once DROP_OK exists), then the next group's. Within
    a group the job order is the original worker's (trial major for paired, branch major for drop). If DROP_FAIL
    exists drop branches are skipped and paired trials still run. Returns when nothing is left or STOP exists."""
    root = ctx.root
    while True:
        if (root / "STOP").exists():
            return
        job = None
        for g in groups:
            done = {m: _done(root, m, "paired.jsonl") for m in g}
            for t in range(N_PAIRED):
                for m in g:
                    key = "trial_%04d" % t
                    if key in done[m] or not _claim(root, "paired_%s_%s" % (m, key)):
                        continue
                    job = ("paired", m, t)
                    break
                if job:
                    break
            if job:
                break
            if (root / "DROP_OK").exists():
                man = load_manifest(ctx.pool)
                done = {m: _done(root, m, "drop.jsonl") for m in g}
                for m, s, b in [(m, s, b) for b in range(N_BRANCHES) for s in range(len(man["states"])) for m in g]:
                    key = "state_%02d_branch_%d" % (s, b)
                    if key in done[m] or not _claim(root, "drop_%s_%s" % (m, key)):
                        continue
                    job = ("drop", m, man, s, b)
                    break
                if job:
                    break
            elif not (root / "DROP_FAIL").exists():
                job = ("wait",)
                break
        if job is None:
            return
        if job[0] == "wait":
            time.sleep(10)
        elif job[0] == "paired":
            paired_trial(ctx, job[1], job[2], video=job[2] in VIDEO_TRIALS)
        else:
            drop_branch(ctx, job[1], job[2], job[3], job[4], video=(job[3], job[4]) in VIDEO_DROPS)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("worker", "paired", "drop", "replay", "repeat", "simreplay"))
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--out", default=str(ROOT))
    p.add_argument("--seed", type=int, default=S.SEED)
    p.add_argument("--method", default="base", choices=tuple(METHODS))
    p.add_argument("--trial-list", type=int, nargs="*", default=[])
    p.add_argument("--states", type=int, nargs="*", default=[])
    p.add_argument("--pairs", nargs="*", default=[], help="state:branch pairs (drop / repeat)")
    p.add_argument("--tag", default="")
    p.add_argument("--pool-dir", default=None, help="saved states for drop/replay (default <out>/pool)")
    p.add_argument("--save-pool-dir", default=None, help="where pi0.5 paired trials save snapshots (default pool dir)")
    p.add_argument("--orig-root", default=None, help="run holding the pool's source trials, for replay (default --out)")
    p.add_argument("--groups", nargs="*", default=None,
                   help="worker method groups in run order, e.g. base,pps task (uses worker_groups)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ctx = Ctx(args)
    logging.info("t59 init states: %d, server %s", len(ctx.init_states), ctx.meta)
    if args.mode == "worker" and args.groups:
        worker_groups(ctx, [tuple(g.split(",")) for g in args.groups])
    elif args.mode == "worker":
        worker(ctx)
    elif args.mode == "paired":
        done = _done(ctx.root, args.method, "paired.jsonl")
        for t in args.trial_list:
            if "trial_%04d" % t not in done:
                paired_trial(ctx, args.method, t)
    elif args.mode in ("drop", "repeat"):
        man = load_manifest(ctx.pool)
        jl = "drop.jsonl" if args.mode == "drop" else "repeat%s.jsonl" % args.tag
        done = _done(ctx.root, args.method, jl)
        for pr in args.pairs:
            s, b = (int(x) for x in pr.split(":"))
            if "state_%02d_branch_%d" % (s, b) not in done:
                drop_branch(ctx, args.method, man, s, b, jl_name=jl, tag=args.tag)
    elif args.mode == "simreplay":
        man = load_manifest(ctx.pool)
        jl = ctx.root / "checks" / "simreplay_check.jsonl"
        jl.parent.mkdir(parents=True, exist_ok=True)
        ok = all([simreplay_state(ctx, man, s, jl) for s in args.states])
        ctx.env.close()
        sys.exit(0 if ok else 3)
    elif args.mode == "replay":
        man = load_manifest(ctx.pool)
        jl = ctx.root / "checks" / "replay_check.jsonl"
        jl.parent.mkdir(parents=True, exist_ok=True)
        ok = all([replay_state(ctx, man, s, jl) for s in args.states])
        ctx.env.close()
        sys.exit(0 if ok else 3)
    ctx.env.close()


if __name__ == "__main__":
    main()
