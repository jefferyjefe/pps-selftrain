"""gym-pusht with an optional goal offset, and a batched closed loop rollout."""

import numpy as np
import torch
from gym_pusht.envs.pusht import PushTEnv

from toy.data import ACT_DIM, CHUNK_DIM, HORIZON, N_ACTION_STEPS, N_OBS, features, unnormalize_xy

BASE_GOAL = np.array([256.0, 256.0, np.pi / 4])
MAX_STEPS = 300  # gym-pusht registers PushT-v0 with max_episode_steps=300


class OffsetGoalPushT(PushTEnv):
    """PushT whose goal pose is translated by a fixed (dx, dy). PushTEnv._setup resets the
    goal on every reset, so the offset is reapplied there."""

    def __init__(self, goal_offset=(0.0, 0.0), **kwargs):
        self.goal_offset = np.asarray(goal_offset, dtype=np.float64)
        super().__init__(obs_type="environment_state_agent_pos", **kwargs)

    def _setup(self):
        super()._setup()
        self.goal_pose = BASE_GOAL + np.array([self.goal_offset[0], self.goal_offset[1], 0.0])


def obs_features(o) -> np.ndarray:
    return features(np.asarray(o["agent_pos"]), np.asarray(o["environment_state"]))


def _block_template():
    env = PushTEnv(obs_type="environment_state_agent_pos")
    env.reset(seed=0)
    env.block.angle = 0.0
    env.block.position = (0.0, 0.0)
    tmpl = PushTEnv.get_keypoints(env._block_shapes)
    cog = np.array(env.block.center_of_gravity)
    env.close()
    return tmpl, cog


_TEMPLATE, _COG = _block_template()


def reset_state_from_frame(agent, kp16) -> np.ndarray:
    """Recover the block pose from its 8 keypoints (2D Procrustes against the T template),
    then convert to the state that PushTEnv.reset(options={"reset_to_state": ...}) expects.
    gym-pusht sets the block position before its angle, and the angle setter rotates about
    the centre of gravity, so the stored position must be pre-compensated."""
    kp = np.asarray(kp16, dtype=np.float64).reshape(8, 2)
    mt, mk = _TEMPLATE.mean(0), kp.mean(0)
    u, _, vt = np.linalg.svd((_TEMPLATE - mt).T @ (kp - mk))
    rot = vt.T @ u.T
    theta = np.arctan2(rot[1, 0], rot[0, 0])
    origin = mk - rot @ mt
    pos = origin - (np.eye(2) - rot) @ _COG
    return np.array([agent[0], agent[1], pos[0], pos[1], theta])


def replay(episode: dict, goal_offset=(0.0, 0.0)) -> dict:
    """Open loop replay of a demo's actions from its first frame."""
    env = OffsetGoalPushT(goal_offset)
    o, _ = env.reset(options={"reset_to_state": reset_state_from_frame(episode["agent"][0], episode["kp"][0])})
    err = [np.abs(o["environment_state"] - episode["kp"][0]).max()]
    cov = []
    for t, a in enumerate(episode["action"]):
        o, _, _, _, info = env.step(a)
        cov.append(info["coverage"])
        if t + 1 < len(episode["kp"]):
            err.append(np.abs(o["environment_state"] - episode["kp"][t + 1]).max())
    env.close()
    return {"max_kp_error": float(max(err)), "final_coverage": float(cov[-1]), "max_coverage": float(max(cov))}


def rollout(policy, seeds, goal_offset=(0.0, 0.0), device="cpu", noise_seed: int = 0,
            max_steps: int = MAX_STEPS) -> dict:
    """Run one episode per env seed in lockstep, batching policy calls across episodes.

    The sampler noise is drawn for every episode at every replan from one generator, so
    two policies evaluated with the same seeds and noise_seed see identical initial
    states and identical noise (common random numbers across conditions).
    """
    n = len(seeds)
    envs = [OffsetGoalPushT(goal_offset) for _ in range(n)]
    hist = []
    for env, s in zip(envs, seeds):
        o, _ = env.reset(seed=int(s))
        hist.append([obs_features(o)] * N_OBS)
    success = np.zeros(n, dtype=bool)
    done = np.zeros(n, dtype=bool)
    steps = np.zeros(n, dtype=int)
    max_cov = np.zeros(n)
    final_cov = np.zeros(n)
    gen = torch.Generator().manual_seed(noise_seed)
    while not done.all():
        noise = torch.randn(n, CHUNK_DIM, generator=gen)
        live = np.where(~done)[0]
        obs = torch.from_numpy(np.stack([np.concatenate(hist[i]) for i in live])).to(device)
        chunk = policy(obs, noise[live].to(device)).float().cpu().numpy()
        chunk = np.clip(unnormalize_xy(chunk.reshape(-1, HORIZON, ACT_DIM)), 0.0, 512.0)
        for j, i in enumerate(live):
            for h in range(N_ACTION_STEPS):
                o, _, terminated, _, info = envs[i].step(chunk[j, h])
                steps[i] += 1
                max_cov[i] = max(max_cov[i], info["coverage"])
                final_cov[i] = info["coverage"]
                hist[i] = hist[i][1:] + [obs_features(o)]
                if terminated:
                    success[i] = done[i] = True
                    break
                if steps[i] >= max_steps:
                    done[i] = True
                    break
    for env in envs:
        env.close()
    return {"success": success, "max_coverage": max_cov, "final_coverage": final_cov, "steps": steps}


PRIMARY_THRESHOLD = 0.90  # primary success threshold. The env's own 0.95 is kept as a secondary metric.


def summarize(res: dict) -> dict:
    """Primary metrics are mean max coverage and success at 0.90 (max coverage over the
    episode above 0.90). success_095 is the env's own termination criterion."""
    mc = res["max_coverage"]
    n = len(mc)
    s90 = (mc > PRIMARY_THRESHOLD).astype(float)
    s95 = res["success"].astype(float)
    binse = lambda p: float(np.sqrt(p * (1 - p) / n))
    return {
        "n": n,
        "mean_max_coverage": float(mc.mean()),
        "mean_max_coverage_se": float(mc.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        "success_090": float(s90.mean()),
        "success_090_se": binse(s90.mean()),
        "success_095": float(s95.mean()),
        "success_095_se": binse(s95.mean()),
        "mean_final_coverage": float(res["final_coverage"].mean()),
        "mean_steps": float(res["steps"].mean()),
    }


def fmt(s: dict) -> str:
    return (f"mean max coverage {s['mean_max_coverage']:.3f} +/- {s['mean_max_coverage_se']:.3f}, "
            f"success@0.90 {s['success_090']:.3f} +/- {s['success_090_se']:.3f}, "
            f"success@0.95 {s['success_095']:.3f} (n={s['n']})")
