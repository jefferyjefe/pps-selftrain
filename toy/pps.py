"""Proxy Policy Steering, Eq. (4) of the paper.

v_pps(x, k, o) = v_base(x, k, o) + gamma * (v_task(x, k, o) - v_ref(x, k, o))

applied at every one of the Euler denoising steps.
"""

import torch

from toy.common import CKPT_DIR
from toy.model import N_DENOISE, euler_sample, k_grid, load_ckpt

MODES = ("pps", "base", "task", "ref")


def load_policies(device, seed: int):
    """Frozen base plus the reference and task proxies trained with the given seed."""
    base = load_ckpt(CKPT_DIR / "base.pt", device)
    ref = load_ckpt(CKPT_DIR / f"seed{seed}" / "ref.pt", device)
    task = load_ckpt(CKPT_DIR / f"seed{seed}" / "task.pt", device)
    return base, ref, task


def velocity_fn(mode: str, base=None, task=None, ref=None, gamma: float = 0.0):
    if mode == "base" or (mode == "pps" and gamma == 0.0):
        return base
    if mode == "task":
        return task
    if mode == "ref":
        return ref
    if mode == "pps":
        def v(x, k, obs):
            return base(x, k, obs) + gamma * (task(x, k, obs) - ref(x, k, obs))
        return v
    raise ValueError(f"unknown mode {mode}")


def make_policy(mode: str, base=None, task=None, ref=None, gamma: float = 0.0, n_steps: int = N_DENOISE):
    """Returns policy(obs, noise) -> normalized action chunk."""
    v = velocity_fn(mode, base, task, ref, gamma)

    def policy(obs, noise):
        return euler_sample(v, obs, noise, n_steps)

    return policy


@torch.no_grad()
def base_denoising_trajectory(base, obs, noise, n_steps: int = N_DENOISE):
    """Roll the frozen base sampler from noise and record (x_k, k, v_base) at each of its
    steps. These are the on-policy distillation targets of Eq. (5)."""
    xs, ks, vs = [], [], []
    x = noise
    grid = k_grid(n_steps)
    for k_cur, k_next in zip(grid[:-1], grid[1:]):
        k = torch.full((x.shape[0],), k_cur, device=x.device)
        v = base(x, k, obs)
        xs.append(x)
        ks.append(k)
        vs.append(v)
        x = x + (k_next - k_cur) * v
    return torch.stack(xs), torch.stack(ks), torch.stack(vs)  # (K, B, D), (K, B), (K, B, D)
