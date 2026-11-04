"""Deliverable 3. Base policy rollouts on t59 with a perturbation at a task event. Runs in the LIBERO venv (Python 3.8).

The episode loop is scripts/m0/eval_libero.py's (10 settle steps, replan every 5 actions, 400 max steps, episode k
uses init state k, env seed seed + k and noise from default_rng((seed, k))), so with --perturbation none and the
same seed an episode is the milestone 0 episode, except that the server now runs the eager sampler. The
perturbation direction is drawn from default_rng((seed, k, 2)). Because the sim and the sampler are deterministic,
a perturbed episode is identical to its unperturbed twin up to the event, which is how the event timing is checked.

One JSONL record per episode (resumable). The record holds the event info from pps_real.perturb, a per step trace
from 5 steps before the event to 25 after, the step at which the bottle is first lifted 2 cm above its start
height, and for perturbed episodes the first step at which the gripper grasps the bottle again.

Usage
  PYTHONPATH=.:/workspace/openpi/third_party/libero /workspace/openpi/examples/libero/.venv/bin/python \
      scripts/stage2/eval_perturb.py --perturbation teleport --episodes 10 --port 8002 \
      --out results/stage2/perturb/teleport.jsonl --video-dir /workspace/runs/stage2/videos
"""

import argparse
import collections
import json
import logging
import os
import pathlib
import time

import imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import websocket_client_policy as _wcp

import pps_real
from pps_real import libero_obs
from pps_real import perturb as _perturb

LIFT_M = 0.02
TRACE_BEFORE, TRACE_AFTER = 5, 25


def done_eps(path):
    out = set()
    if os.path.exists(path):
        for line in open(path):
            out.add(json.loads(line)["episode"])
    return out


def border(img, color):
    img = img.copy()
    img[:6], img[-6:], img[:, :6], img[:, -6:] = color, color, color, color
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--event", default="grasp", choices=_perturb.EVENTS)
    p.add_argument("--perturbation", required=True, choices=_perturb.PERTURBATIONS)
    p.add_argument("--teleport-dist", type=float, default=0.05)
    p.add_argument("--release-frames", type=int, default=3)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--ep-start", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8002)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--out", required=True)
    p.add_argument("--video-dir", default=None)
    p.add_argument("--videos", type=int, default=2,
                   help="keep videos of the first N episodes in which the event fires (counted over --video-dir)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    suite = benchmark.get_benchmark_dict()[pps_real.T59_SUITE]()
    task = suite.get_task(pps_real.T59_TASK_ID)
    assert task.name == pps_real.T59_NAME
    init_states = suite.get_task_init_states(pps_real.T59_TASK_ID)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    client = _wcp.WebsocketClientPolicy(args.host, args.port)
    server_meta = client.get_server_metadata().get("m0", {})
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    finished = done_eps(args.out)
    max_steps = libero_obs.MAX_STEPS_LIBERO_90
    tag = args.perturbation if args.perturbation == "none" else f"{args.event}_{args.perturbation}"

    for ep in range(args.ep_start, args.ep_start + args.episodes):
        if ep in finished:
            continue
        t_start = time.monotonic()
        env.seed(args.seed + ep)
        env.reset()
        obs = env.set_init_state(init_states[ep % len(init_states)])
        pert = _perturb.Perturber(env, event=args.event, perturbation=args.perturbation,
                                  rng=np.random.default_rng((args.seed, ep, 2)), teleport_dist=args.teleport_dist,
                                  release_frames=args.release_frames)
        noise_rng = np.random.default_rng((args.seed, ep))
        plan = collections.deque()
        record_video = args.video_dir is not None  # frames kept in memory, written only if the event fired
        frames, trace, infer_ms = [], [], []
        mark_left = 0
        done, t, err = False, 0, None
        while t < max_steps + libero_obs.SETTLE_STEPS:
            try:
                if t < libero_obs.SETTLE_STEPS:
                    obs, _, done, _ = env.step(libero_obs.DUMMY_ACTION)
                    t += 1
                    if t == libero_obs.SETTLE_STEPS:
                        pert.start()
                    continue
                img, wrist = libero_obs.process_images(obs)
                if record_video:
                    f = np.concatenate([img, wrist], axis=1)
                    if mark_left > 0 or pert.overriding:
                        f = border(f, (255, 0, 0))
                        mark_left = max(0, mark_left - 1)
                    frames.append(f)
                if not plan:
                    el = {"observation/image": img, "observation/wrist_image": wrist,
                          "observation/state": libero_obs.policy_state(obs), "prompt": str(task.language),
                          "__noise": noise_rng.standard_normal(libero_obs.NOISE_SHAPE).astype(np.float32)}
                    out = client.infer(el)
                    infer_ms.append(out["server_stats"]["infer_ms"])
                    plan.extend(out["actions"][: args.replan_steps])
                action = pert.filter_action(plan.popleft().tolist())
                obs, _, done, _ = env.step(action)
                trace.append({"t": t, "a6": float(action[6]), "grip": [float(v) for v in obs["robot0_gripper_qpos"]],
                              "w": pert.width(), "tw": pert.target_width(),
                              "obj": [float(v) for v in pert.obj_pos()], "grasp": pert.grasping()})
                if done:
                    break
                obs_pre = obs
                obs, fired = pert.after_step(t, action, obs)
                if fired:
                    trace[-1]["obj_after_perturb"] = [float(v) for v in pert.obj_pos()]
                    mark_left = 5 if args.perturbation == "teleport" else 0
                    if record_video:
                        pathlib.Path(args.video_dir).mkdir(parents=True, exist_ok=True)
                        im1, wr1 = libero_obs.process_images(obs_pre)
                        imageio.imwrite(f"{args.video_dir}/t59_{tag}_ep{ep}_event.png", np.concatenate([im1, wr1], 1))
                        im2, wr2 = libero_obs.process_images(obs)
                        imageio.imwrite(f"{args.video_dir}/t59_{tag}_ep{ep}_after.png", np.concatenate([im2, wr2], 1))
                t += 1
            except Exception as e:  # same policy as openpi main.py, but recorded
                err = repr(e)
                logging.error("episode error %s", err)
                break
        ev = pert.fired_step
        z0 = pert.z0
        lift = next((r["t"] for r in trace if r["obj"][2] - z0 > LIFT_M), None)
        regrasp = None
        if ev is not None and args.perturbation != "none":
            after = ev + (args.release_frames if args.perturbation == "release" else 0)
            # first grasp after the gripper lost the bottle
            lost = next((r["t"] for r in trace if r["t"] > ev and not r["grasp"]), None)
            if lost is not None:
                regrasp = next((r["t"] for r in trace if r["t"] > max(lost, after) and r["grasp"]), None)
        win = [r for r in trace if ev is not None and ev - TRACE_BEFORE <= r["t"] <= ev + TRACE_AFTER]
        rec = {
            "suite": pps_real.T59_SUITE, "task_id": pps_real.T59_TASK_ID, "episode": ep, "seed": args.seed,
            "event": args.event, "perturbation": args.perturbation, "teleport_dist": args.teleport_dist,
            "release_frames": args.release_frames, "success": bool(done), "env_steps": t, "error": err,
            "fired": ev is not None, "event_info": pert.info, "obj_start_z": z0,
            "lift_step": lift,
            # the event predicate recomputed from the logged trace, independently of the live detector
            "first_event_step_in_trace": next((b["t"] for a, b in zip(trace, trace[1:])
                                               if _perturb.closes_on(b["grasp"], b["a6"], b["w"], a["w"], b["tw"])),
                                              None),
            "first_grasp_step": next((r["t"] for r in trace if r["grasp"]), None),
            "regrasp_step": regrasp, "trace_window": win, "policy_calls": len(infer_ms),
            "infer_ms_mean": float(np.mean(infer_ms[1:])) if len(infer_ms) > 1 else None,
            "wall_s": time.monotonic() - t_start, "server": server_meta,
        }
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        logging.info("ep %d %s fired=%s at %s success=%s steps=%d wall=%.1fs", ep, tag, ev is not None, ev, done, t,
                     rec["wall_s"])
        n_saved = len(list(pathlib.Path(args.video_dir).glob(f"t59_{tag}_ep*.mp4"))) if args.video_dir else 0
        if frames and ev is not None and n_saved < args.videos:
            imageio.mimwrite(f"{args.video_dir}/t59_{tag}_ep{ep}_{'ok' if done else 'fail'}.mp4", frames, fps=20)
    env.close()


if __name__ == "__main__":
    main()
