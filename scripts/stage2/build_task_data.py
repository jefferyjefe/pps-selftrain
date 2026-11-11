"""Build the task proxy's training targets: the t59 demo action chunks in pi0.5's own action space.

Frames are the 5571 frames of the 50 t59 demos, in the same order as /workspace/data/t59_distill and
/workspace/data/t59_demos/policy_obs.npz. Raw 7-d actions come from /workspace/data/t59_demos/states.npz
(export_demo_states.py, every hdf5 step, no no-op filtering, matching the distillation frames).

Nothing about the action space is hand-rolled.
- Chunking: openpi's create_torch_dataset builds a LeRobotDataset with delta_timestamps
  {"actions": [t / fps for t in range(action_horizon)]}. The chunk indices here come from LeRobot's own
  get_delta_indices and LeRobotDataset._get_query_indices, run on these episodes' boundaries. Near the end of an
  episode LeRobot clamps indices to the last frame (the last action is repeated) and reports actions_is_pad, which
  openpi's loss ignores. The is_pad mask is saved for reference only.
- Transforms: openpi's data_loader.transform_dataset with the pi05_libero DataConfig (repack, LiberoInputs,
  Normalize with use_quantile_norm, model transforms incl. PadStatesAndActions to 32), the norm stats being the
  checkpoint's own (checkpoints.load_norm_stats(<ckpt>/assets, asset_id)), as openpi's policy_config does.
  The output "actions" (10, 32) are taken verbatim.

Output /workspace/data/t59_task/: actions.npy (F, 10, 32) float32 normalized and padded (the training target),
raw_chunks.npy (F, 10, 7) float32, actions_is_pad.npy (F, 10) bool, demo_idx.npy, frame_idx.npy, manifest.json.

Usage
  /workspace/openpi/.venv/bin/python scripts/stage2/build_task_data.py
"""

import argparse
import dataclasses
import json
import pathlib
import types

import numpy as np
import torch
from lerobot.common.datasets import lerobot_dataset
from lerobot.common.datasets.utils import get_delta_indices

from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

LIBERO_FPS = 10  # physical-intelligence/libero meta fps. Only scales delta_timestamps back to integer offsets.


class DemoFrames(torch.utils.data.Dataset):
    """Map style dataset yielding what openpi's LIBERO LeRobotDataset yields (after PromptFromLeRobotTask)."""

    def __init__(self, obs, raw_actions, demo_idx, action_horizon):
        self.obs, self.raw, self.demo = obs, raw_actions, demo_idx
        starts = np.r_[0, np.nonzero(np.diff(demo_idx))[0] + 1]
        ends = np.r_[starts[1:], len(demo_idx)]
        self.ep_of = np.repeat(np.arange(len(starts)), ends - starts)
        # LeRobot internals: episode_data_index and delta_indices, as LeRobotDataset sets them up.
        self.lr = types.SimpleNamespace(
            episode_data_index={"from": torch.tensor(starts), "to": torch.tensor(ends)},
            delta_indices=get_delta_indices(
                {"actions": [t / LIBERO_FPS for t in range(action_horizon)]}, LIBERO_FPS))

    def __len__(self):
        return len(self.demo)

    def query(self, idx):
        q, pad = lerobot_dataset.LeRobotDataset._get_query_indices(self.lr, idx, int(self.ep_of[idx]))
        return q["actions"], pad["actions_is_pad"].numpy()

    def __getitem__(self, idx):
        q, _ = self.query(idx)
        return {"image": self.obs["image"][idx], "wrist_image": self.obs["wrist_image"][idx],
                "state": self.obs["state"][idx], "actions": self.raw[q], "prompt": str(self.obs["prompt"])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="pi05_libero")
    p.add_argument("--ckpt", default="/workspace/checkpoints/pi05_libero_pytorch")
    p.add_argument("--states", default="/workspace/data/t59_demos/states.npz")
    p.add_argument("--obs", default="/workspace/data/t59_demos/policy_obs.npz")
    p.add_argument("--distill", default="/workspace/data/t59_distill")
    p.add_argument("--out", default="/workspace/data/t59_task")
    args = p.parse_args()

    tc = _config.get_config(args.config)
    data_config = tc.data.create(tc.assets_dirs, tc.model)
    norm_stats = _checkpoints.load_norm_stats(pathlib.Path(args.ckpt) / "assets", data_config.asset_id)
    data_config = dataclasses.replace(data_config, norm_stats=norm_stats)
    H = tc.model.action_horizon
    assert H == 10 and tc.model.action_dim == 32 and data_config.use_quantile_norm
    assert tuple(data_config.action_sequence_keys) == ("actions",)

    st = np.load(args.states)
    obs = dict(np.load(args.obs))
    dd = pathlib.Path(args.distill)
    demo, frame = np.load(dd / "demo_idx.npy"), np.load(dd / "frame_idx.npy")
    for z in (st, obs):
        assert np.array_equal(z["demo_idx"], demo) and np.array_equal(z["frame_idx"], frame)
    # contiguous episodes, frames 0..n-1 in order (LeRobot's index layout)
    assert np.all(np.diff(demo) >= 0)
    starts = np.r_[0, np.nonzero(np.diff(demo))[0] + 1]
    assert np.array_equal(frame, np.arange(len(frame)) - np.repeat(starts, np.diff(np.r_[starts, len(frame)])))

    raw = st["actions"].astype(np.float32)
    ds = DemoFrames(obs, raw, demo, H)
    tds = _data_loader.transform_dataset(ds, data_config)

    F = len(ds)
    actions = np.zeros((F, H, tc.model.action_dim), np.float32)
    raw_chunks = np.zeros((F, H, raw.shape[1]), np.float32)
    is_pad = np.zeros((F, H), bool)
    for i in range(F):
        q, pad = ds.query(i)
        raw_chunks[i], is_pad[i] = raw[q], pad
        a = np.asarray(tds[i]["actions"])
        assert a.shape == (H, tc.model.action_dim), a.shape
        actions[i] = a
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "actions.npy", actions)
    np.save(out / "raw_chunks.npy", raw_chunks)
    np.save(out / "actions_is_pad.npy", is_pad)
    np.save(out / "demo_idx.npy", demo)
    np.save(out / "frame_idx.npy", frame)
    ns = norm_stats["actions"]
    man = dict(
        n_frames=F, action_horizon=H, action_dim=tc.model.action_dim, config=args.config,
        norm_stats=str(pathlib.Path(args.ckpt) / "assets" / data_config.asset_id),
        use_quantile_norm=bool(data_config.use_quantile_norm),
        actions_q01=np.asarray(ns.q01).tolist(), actions_q99=np.asarray(ns.q99).tolist(),
        transforms=[type(t).__name__ for t in tds._transforms] if hasattr(tds, "_transforms") else None,
        chunking="LeRobotDataset._get_query_indices: indices clamped to the episode, last action repeated past the "
                 "end, actions_is_pad saved but (as in openpi) not used by the loss",
        n_chunks_with_padding=int(is_pad.any(1).sum()),
        padded_dims_all_zero=bool(np.all(actions[..., 7:] == 0)),
        source_actions=args.states, frames=args.distill,
    )
    (out / "manifest.json").write_text(json.dumps(man, indent=2))
    print(json.dumps({k: v for k, v in man.items() if k not in ("actions_q01", "actions_q99")}, indent=2))


if __name__ == "__main__":
    main()
