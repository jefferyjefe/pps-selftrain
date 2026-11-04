"""PushT demonstrations as state observations.

lerobot/pusht stores only the agent xy plus 96x96 images, so the block pose is not in its
state. lerobot/pusht_keypoints is the same 206 episodes (identical index, agent state and
actions) with the 8 T-block keypoints added. We take actions and agent state from
lerobot/pusht and join the keypoints from lerobot/pusht_keypoints, asserting they align.
"""

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from toy.common import DATA_DIR

# Pinned dataset revisions (commit hashes on the Hugging Face Hub).
DATASETS = {
    "lerobot/pusht": "7628202a2180972f291ba1bc6723834921e72c19",
    "lerobot/pusht_keypoints": "ace8c161a68bc025c21a5f29f85b86a9a2c5e64b",
}

N_OBS = 2  # observation history (current and previous frame)
HORIZON = 16  # action chunk length
N_ACTION_STEPS = 8  # executed before replanning
ACT_DIM = 2
FEAT_DIM = 2 + 16  # agent xy + 8 keypoints xy
OBS_DIM = N_OBS * FEAT_DIM
CHUNK_DIM = HORIZON * ACT_DIM


def normalize_xy(x):
    """Pixel coordinates in [0, 512] to roughly [-1, 1]. Same map for obs and actions."""
    return (x - 256.0) / 256.0


def unnormalize_xy(x):
    return x * 256.0 + 256.0


def _read(repo_id: str) -> dict:
    local = DATA_DIR / "hf" / repo_id.split("/")[1]
    snapshot_download(
        repo_id,
        repo_type="dataset",
        revision=DATASETS[repo_id],
        allow_patterns=["data/*", "meta/*"],
        local_dir=local,
    )
    files = sorted((local / "data").rglob("*.parquet"))
    cols = {}
    for f in files:
        t = pq.read_table(f)
        for c in t.column_names:
            cols.setdefault(c, []).append(t.column(c).to_numpy(zero_copy_only=False))
    return {c: np.concatenate(v) for c, v in cols.items()}


def load_pusht_episodes() -> list[dict]:
    """Returns a list of episodes, each with agent (T,2), kp (T,16), action (T,2) in pixels."""
    a = _read("lerobot/pusht")
    b = _read("lerobot/pusht_keypoints")
    stack = lambda x: np.stack(x).astype(np.float32)
    assert np.array_equal(a["index"], b["index"])
    assert np.array_equal(stack(a["observation.state"]), stack(b["observation.state"]))
    assert np.array_equal(stack(a["action"]), stack(b["action"]))
    agent = stack(a["observation.state"])
    action = stack(a["action"])
    kp = stack(b["observation.environment_state"])
    ep = a["episode_index"]
    episodes = []
    for e in np.unique(ep):
        idx = np.where(ep == e)[0]
        assert np.all(np.diff(a["frame_index"][idx]) == 1)
        episodes.append({"episode": int(e), "agent": agent[idx], "kp": kp[idx], "action": action[idx]})
    return episodes


def translate_episode(ep: dict, offset) -> dict:
    """Rigidly translate a whole episode. PushT physics is translation invariant away from walls."""
    d = np.asarray(offset, dtype=np.float32)
    return {
        **ep,
        "agent": ep["agent"] + d,
        "kp": ep["kp"] + np.tile(d, 8),
        "action": ep["action"] + d,
    }


def features(agent, kp):
    """Per frame feature vector, normalized. agent (...,2), kp (...,16)."""
    return normalize_xy(np.concatenate([agent, kp], axis=-1)).astype(np.float32)


def make_windows(episodes: list[dict]):
    """Training pairs. obs is frames [t-1, t] (t-1 clamped at 0); actions are [t, t+H) padded with the last action."""
    obs, act = [], []
    for ep in episodes:
        f = features(ep["agent"], ep["kp"])
        a = normalize_xy(ep["action"]).astype(np.float32)
        T = len(f)
        for t in range(T):
            hist = [f[max(t - i, 0)] for i in reversed(range(N_OBS))]
            obs.append(np.concatenate(hist))
            idx = np.minimum(np.arange(t, t + HORIZON), T - 1)
            act.append(a[idx].reshape(-1))
    return np.stack(obs), np.stack(act)


def save_episodes(episodes: list[dict], path, **meta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        agent=np.concatenate([e["agent"] for e in episodes]),
        kp=np.concatenate([e["kp"] for e in episodes]),
        action=np.concatenate([e["action"] for e in episodes]),
        lengths=np.array([len(e["agent"]) for e in episodes]),
        source_episode=np.array([e["episode"] for e in episodes]),
        **meta,
    )


def load_episodes(path) -> tuple[list[dict], dict]:
    z = np.load(path)
    bounds = np.concatenate([[0], np.cumsum(z["lengths"])])
    eps = [
        {
            "episode": int(z["source_episode"][i]),
            "agent": z["agent"][s:e],
            "kp": z["kp"][s:e],
            "action": z["action"][s:e],
        }
        for i, (s, e) in enumerate(zip(bounds[:-1], bounds[1:]))
    ]
    meta = {k: z[k] for k in z.files if k not in {"agent", "kp", "action", "lengths", "source_episode"}}
    return eps, meta


VARIANT_DEMOS = DATA_DIR / "toy" / "variant_demos.npz"


def variant_offset() -> tuple[float, float]:
    """Goal offset chosen by make_task.py, stored with the variant demos."""
    _, meta = load_episodes(VARIANT_DEMOS)
    return tuple(float(v) for v in meta["goal_offset"])
