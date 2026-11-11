"""Inputs for the faithful proxies, from the same 50 t59 demos and the same openpi pi05_libero pipeline as before.

Reruns scripts/stage2/build_task_data.py's dataset (DemoFrames: LeRobot chunking on the demo episodes) through openpi's
transform_dataset with the pi05_libero DataConfig and the checkpoint's own norm stats, and keeps the model inputs the
proxies need besides the images:
  state_norm.npy   (F, 32) float32  the normalized state pi0.5's Observation carries (quantile norm, padded to 32).
                   The proxies' state token reads its first STATE_DIM entries (8, from the norm stats).
  tokenized_prompt.npy, tokenized_prompt_mask.npy  (L,)  the teacher's prompt tokens (identical for every frame).
  real_dims.json   action and state dims read from the norm stats, and the image keys and masks.
Asserts: images out of the pipeline equal /workspace/data/t59_demos/policy_obs.npz frames, the right wrist slot is
zero and masked, the action targets equal /workspace/data/t59_task/actions.npy (so the task proxy trains on exactly the
v3 targets, real dims only), the prompt tokens are the same for every frame, and padding dims past the real ones are 0.

Usage: /workspace/openpi/.venv/bin/python scripts/faithful/build_faithful_data.py
"""

import dataclasses
import json
import pathlib
import sys

import numpy as np

from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "stage2"))
from build_task_data import DemoFrames  # noqa: E402

CKPT = "/workspace/checkpoints/pi05_libero_pytorch"
OBS = "/workspace/data/t59_demos/policy_obs.npz"
STATES = "/workspace/data/t59_demos/states.npz"
TASK = pathlib.Path("/workspace/data/t59_task")
OUT = pathlib.Path("/workspace/data/t59_faithful")


def main():
    tc = _config.get_config("pi05_libero")
    data_config = tc.data.create(tc.assets_dirs, tc.model)
    norm_stats = _checkpoints.load_norm_stats(pathlib.Path(CKPT) / "assets", data_config.asset_id)
    data_config = dataclasses.replace(data_config, norm_stats=norm_stats)
    A = len(np.asarray(norm_stats["actions"].q01))
    S = len(np.asarray(norm_stats["state"].q01))
    st = np.load(STATES)
    obs = dict(np.load(OBS))
    demo = np.load(TASK / "demo_idx.npy")
    raw = st["actions"].astype(np.float32)
    assert raw.shape[1] == A, (raw.shape, A)
    ds = DemoFrames(obs, raw, demo, tc.model.action_horizon)
    tds = _data_loader.transform_dataset(ds, data_config)
    actions_v3 = np.load(TASK / "actions.npy")
    F = len(ds)
    state = np.zeros((F, tc.model.action_dim), np.float32)
    tok = tokm = None
    for i in range(F):
        d = tds[i]
        if i < 64 or i % 97 == 0:
            assert np.array_equal(d["image"]["base_0_rgb"], obs["image"][i])
            assert np.array_equal(d["image"]["left_wrist_0_rgb"], obs["wrist_image"][i])
            assert not np.any(d["image"]["right_wrist_0_rgb"]) and not bool(d["image_mask"]["right_wrist_0_rgb"])
            assert bool(d["image_mask"]["base_0_rgb"]) and bool(d["image_mask"]["left_wrist_0_rgb"])
        assert np.array_equal(np.asarray(d["actions"]).astype(np.float32), actions_v3[i])  # stored as float32 by build_task_data
        state[i] = np.asarray(d["state"])
        if tok is None:
            tok, tokm = np.asarray(d["tokenized_prompt"]), np.asarray(d["tokenized_prompt_mask"])
        else:
            assert np.array_equal(tok, d["tokenized_prompt"]) and np.array_equal(tokm, d["tokenized_prompt_mask"])
    assert np.all(state[:, S:] == 0) and np.all(actions_v3[..., A:] == 0)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "state_norm.npy", state)
    np.save(OUT / "tokenized_prompt.npy", tok)
    np.save(OUT / "tokenized_prompt_mask.npy", tokm)
    info = dict(n_frames=F, action_dim_real=A, state_dim_real=S, model_action_dim=tc.model.action_dim,
                action_horizon=tc.model.action_horizon, use_quantile_norm=bool(data_config.use_quantile_norm),
                norm_stats=str(pathlib.Path(CKPT) / "assets" / data_config.asset_id),
                prompt=str(obs["prompt"]), n_prompt_tokens_valid=int(tokm.sum()), max_token_len=int(len(tok)),
                image_keys_base=["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb (zeros, masked)"],
                image_keys_proxy=["base_0_rgb", "left_wrist_0_rgb"],
                actions_target="/workspace/data/t59_task/actions.npy[..., :action_dim_real] (asserted equal to the "
                               "pipeline's output for every frame)")
    (OUT / "real_dims.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
