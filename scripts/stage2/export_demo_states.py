"""Export sim states, actions and the low dim obs of the 50 t59 demos from the LIBERO hdf5 to npz.

The LIBERO client venv (Python 3.8) has no h5py, so this step runs in the openpi venv and the
renderer reads the npz.

Usage
  /workspace/openpi/.venv/bin/python scripts/stage2/export_demo_states.py
"""

import argparse
import json
import pathlib

import h5py
import numpy as np

import pps_real


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5", default=pps_real.T59_DEMO_HDF5)
    p.add_argument("--out", default="/workspace/data/t59_demos/states.npz")
    args = p.parse_args()
    f = h5py.File(args.hdf5, "r")
    d = f["data"]
    keys = sorted(d.keys(), key=lambda s: int(s.split("_")[1]))
    states, actions, ee_pos, gripper, demo_idx, frame_idx, img128, wrist128 = [], [], [], [], [], [], [], []
    for i, k in enumerate(keys):
        g = d[k]
        n = g["actions"].shape[0]
        states.append(g["states"][()])
        actions.append(g["actions"][()])
        ee_pos.append(g["obs"]["ee_pos"][()])
        gripper.append(g["obs"]["gripper_states"][()])
        img128.append(g["obs"]["agentview_rgb"][()])
        wrist128.append(g["obs"]["eye_in_hand_rgb"][()])
        demo_idx.append(np.full(n, i, np.int32))
        frame_idx.append(np.arange(n, dtype=np.int32))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, states=np.concatenate(states), actions=np.concatenate(actions), ee_pos=np.concatenate(ee_pos),
             gripper_states=np.concatenate(gripper), demo_idx=np.concatenate(demo_idx),
             frame_idx=np.concatenate(frame_idx), demo_keys=np.array(keys),
             agentview_rgb_128=np.concatenate(img128), eye_in_hand_rgb_128=np.concatenate(wrist128))
    meta = {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in d.attrs.items()}
    meta["n_frames"] = int(sum(len(s) for s in states))
    (out.parent / "demo_attrs.json").write_text(json.dumps(meta, indent=1))
    print(out, meta["n_frames"], "frames from", len(keys), "demos")


if __name__ == "__main__":
    main()
