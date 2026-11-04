"""Render policy observations for every frame of the 50 t59 demos. Runs in the LIBERO venv (Python 3.8).

Each recorded MuJoCo state is set in the same 256 px OffScreenRenderEnv the eval uses, and the images are
processed exactly as the eval client processes them (pps_real.libero_obs). The LIBERO hdf5 stores only
128 px images, which are not what the base sees at eval time.

Fidelity checks written to the output json.
- eef position from the regenerated obs against the recorded ee_pos.
- a second env at 128 px renders the same states, compared pixel wise to the stored 128 px images.

Usage
  PYTHONPATH=.:/workspace/openpi/third_party/libero /workspace/openpi/examples/libero/.venv/bin/python \
      scripts/stage2/render_demo_obs.py
"""

import argparse
import json
import pathlib
import time

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

import pps_real
from pps_real import libero_obs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--states", default="/workspace/data/t59_demos/states.npz")
    p.add_argument("--out", default="/workspace/data/t59_demos/policy_obs.npz")
    p.add_argument("--check-every", type=int, default=25, help="128 px pixel check on every Nth frame")
    args = p.parse_args()
    z = np.load(args.states)
    states = z["states"]
    suite = benchmark.get_benchmark_dict()[pps_real.T59_SUITE]()
    task = suite.get_task(pps_real.T59_TASK_ID)
    assert task.name == pps_real.T59_NAME, task.name
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(0)
    env.reset()
    env128 = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=128, camera_widths=128)
    env128.seed(0)
    env128.reset()
    n_state = env.sim.get_state().flatten().shape[0]
    assert n_state == states.shape[1], (n_state, states.shape)

    t0 = time.time()
    N = len(states)
    images = np.zeros((N, 224, 224, 3), np.uint8)
    wrists = np.zeros((N, 224, 224, 3), np.uint8)
    pstate = np.zeros((N, 8), np.float32)
    eef_err = np.zeros(N, np.float32)
    pix = []
    for i in range(N):
        obs = env.set_init_state(states[i])
        images[i], wrists[i] = libero_obs.process_images(obs)
        pstate[i] = libero_obs.policy_state(obs)
        eef_err[i] = np.abs(obs["robot0_eef_pos"] - z["ee_pos"][i]).max()
        if i % args.check_every == 0:
            o128 = env128.set_init_state(states[i])
            # LIBERO hdf5 images use the opengl convention, the same orientation as raw env renders.
            a = o128["agentview_image"].astype(np.float32)
            w = o128["robot0_eye_in_hand_image"].astype(np.float32)
            pix.append((int(i), float(np.abs(a - z["agentview_rgb_128"][i]).mean()),
                        float(np.abs(w - z["eye_in_hand_rgb_128"][i]).mean())))
        if i % 500 == 0:
            print(i, N, f"{time.time() - t0:.0f}s", flush=True)
    np.savez(args.out, image=images, wrist_image=wrists, state=pstate, demo_idx=z["demo_idx"],
             frame_idx=z["frame_idx"], prompt=np.array(task.language))
    pix = np.array(pix)
    report = {
        "n_frames": int(N), "prompt": task.language, "render_s": time.time() - t0,
        "eef_pos_max_abs_err_m": float(eef_err.max()),
        "pixel_check_frames": int(len(pix)),
        "agentview_128_mean_abs_diff": [float(pix[:, 1].mean()), float(pix[:, 1].max())],
        "wrist_128_mean_abs_diff": [float(pix[:, 2].mean()), float(pix[:, 2].max())],
        "pixel_diff_units": "mean abs difference in 0..255 per pixel channel, [mean over checked frames, max]",
    }
    pathlib.Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    env.close()
    env128.close()


if __name__ == "__main__":
    main()
