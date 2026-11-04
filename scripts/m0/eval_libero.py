"""LIBERO eval client for milestone 0. Runs inside openpi's examples/libero/.venv (Python 3.8).

Follows openpi examples/libero/main.py (10 settle steps, replan every 5 actions, 224 px images,
same max steps per suite) with these changes.
- Episode k uses LIBERO init state k, env seed (seed + k) and flow matching noise from
  numpy default_rng((seed, k)), so each episode is reproducible on its own and independent of order.
- Selected task ids only, results appended per episode to a JSONL file (resumable).
- Optional object placement offset at reset (see offsets.py), used by experiment 3.

Usage
  python scripts/m0/eval_libero.py --suite libero_90 --task-ids 3 7 --episodes 30 --port 8000 --out results/raw/x.jsonl
"""

import argparse
import collections
import json
import logging
import math
import os
import pathlib
import sys
import time

import imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _wcp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import offsets  # noqa: E402

DUMMY = [0.0] * 6 + [-1.0]
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}
NOISE_SHAPE = (10, 32)  # pi05_libero action_horizon x action_dim


def quat2axisangle(quat):
    quat = quat.copy()
    quat[3] = min(max(quat[3], -1.0), 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def done_keys(path):
    keys = set()
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            keys.add((r["suite"], r["task_id"], r["episode"], r.get("offset_tag", "")))
    return keys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", required=True)
    p.add_argument("--task-ids", type=int, nargs="+", required=True)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--ep-start", type=int, default=0, help="first init state index")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--out", required=True)
    p.add_argument("--video-dir", default=None, help="save the first --videos episodes per task here")
    p.add_argument("--videos", type=int, default=2)
    p.add_argument("--offset", type=str, default=None, help="offset spec for offsets.apply, e.g. 'bowl:0.00,0.05'")
    p.add_argument("--offset-tag", type=str, default="")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    suite = benchmark.get_benchmark_dict()[args.suite]()
    max_steps = MAX_STEPS[args.suite]
    client = _wcp.WebsocketClientPolicy(args.host, args.port)
    server_meta = client.get_server_metadata().get("m0", {})
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    finished = done_keys(args.out)

    for task_id in args.task_ids:
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        prompt = task.language
        for ep in range(args.ep_start, args.ep_start + args.episodes):
            if (args.suite, task_id, ep, args.offset_tag) in finished:
                continue
            t_start = time.monotonic()
            env.seed(args.seed + ep)
            env.reset()
            obs = env.set_init_state(init_states[ep % len(init_states)])
            offset_info = None
            if args.offset:
                obs, offset_info = offsets.apply(env, args.offset, np.random.default_rng((args.seed, ep, 1)))
            noise_rng = np.random.default_rng((args.seed, ep))
            plan = collections.deque()
            frames, infer_ms, n_calls = [], [], 0
            peak_alloc = peak_res = 0.0
            done, t, err = False, 0, None
            while t < max_steps + 10:
                try:
                    if t < 10:
                        obs, _, done, _ = env.step(DUMMY)
                        t += 1
                        continue
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
                    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224))
                    if args.video_dir and ep - args.ep_start < args.videos:
                        frames.append(img)
                    if not plan:
                        el = {
                            "observation/image": img,
                            "observation/wrist_image": wrist,
                            "observation/state": np.concatenate(
                                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                            ),
                            "prompt": str(prompt),
                            "__noise": noise_rng.standard_normal(NOISE_SHAPE).astype(np.float32),
                        }
                        out = client.infer(el)
                        st = out["server_stats"]
                        infer_ms.append(st["infer_ms"])
                        peak_alloc = max(peak_alloc, st["max_mem_allocated_gb"])
                        peak_res = max(peak_res, st["max_mem_reserved_gb"])
                        n_calls += 1
                        plan.extend(out["actions"][: args.replan_steps])
                    obs, _, done, _ = env.step(plan.popleft().tolist())
                    if done:
                        break
                    t += 1
                except Exception as e:  # same policy as openpi main.py, but recorded
                    err = repr(e)
                    logging.error("episode error %s", err)
                    break
            wall = time.monotonic() - t_start
            rec = {
                "suite": args.suite, "task_id": task_id, "task": task.name, "prompt": prompt, "episode": ep,
                "seed": args.seed, "success": bool(done), "env_steps": t, "wall_s": wall, "policy_calls": n_calls,
                "infer_ms_mean": float(np.mean(infer_ms[1:])) if len(infer_ms) > 1 else None,
                "infer_ms_first": infer_ms[0] if infer_ms else None,
                "peak_mem_allocated_gb": peak_alloc, "peak_mem_reserved_gb": peak_res,
                "num_denoise_steps": server_meta.get("num_denoise_steps"), "checkpoint": server_meta.get("checkpoint"),
                "norm_stats": server_meta.get("norm_stats"), "offset": args.offset, "offset_tag": args.offset_tag,
                "offset_info": offset_info, "error": err,
            }
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            logging.info("task %d ep %d success=%s steps=%d wall=%.1fs", task_id, ep, done, t, wall)
            if frames:
                pathlib.Path(args.video_dir).mkdir(parents=True, exist_ok=True)
                tag = f"{args.offset_tag}_" if args.offset_tag else ""
                imageio.mimwrite(f"{args.video_dir}/{args.suite}_t{task_id}_{tag}ep{ep}_{'ok' if done else 'fail'}.mp4",
                                 frames, fps=10)
        env.close()


if __name__ == "__main__":
    main()
