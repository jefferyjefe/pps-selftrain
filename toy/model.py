"""Flow matching velocity network and losses.

Convention follows the PPS paper (and pi0). x_k = k * eps + (1 - k) * a, with k = 1 pure
noise and k = 0 the clean action. The network regresses the velocity eps - a. Sampling
starts at k = 1 and takes K Euler steps to k = 0.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from toy.data import CHUNK_DIM, OBS_DIM

N_DENOISE = 10  # Euler steps, as in the paper (Appendix A.2)


def sinusoidal(k: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=k.device) / half)
    ang = (k * 1000.0)[:, None] * freqs[None]
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.cond = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, h, c):
        return h + self.fc2(F.gelu(self.fc1(self.norm(h)) + self.cond(c)))


class VelocityNet(nn.Module):
    """Residual MLP. The noised chunk is fused with a sinusoidal time embedding by
    concatenation and an MLP (pi0 style); the observation enters every block."""

    def __init__(self, width: int = 256, depth: int = 3, temb_dim: int = 64,
                 obs_dim: int = OBS_DIM, act_dim: int = CHUNK_DIM):
        super().__init__()
        self.config = dict(width=width, depth=depth, temb_dim=temb_dim, obs_dim=obs_dim, act_dim=act_dim)
        self.temb_dim = temb_dim
        self.act_in = nn.Sequential(nn.Linear(act_dim + temb_dim, width), nn.GELU(), nn.Linear(width, width))
        self.obs_in = nn.Sequential(nn.Linear(obs_dim, width), nn.GELU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([ResBlock(width) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, act_dim))

    def forward(self, x, k, obs):
        k = k.expand(x.shape[0]) if k.dim() == 0 else k
        h = self.act_in(torch.cat([x, sinusoidal(k, self.temb_dim)], dim=-1))
        c = self.obs_in(obs)
        for blk in self.blocks:
            h = blk(h, c)
        return self.out(h)


def n_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


def sample_k(n: int, device) -> torch.Tensor:
    """Shifted Beta(1.5, 1) as in pi0 and the PPS paper. Mass leans toward noisier k."""
    k = torch.distributions.Beta(1.5, 1.0).sample((n,))
    return (0.001 + 0.999 * k).to(device)


def fm_loss(net, obs, a):
    """Standard flow matching loss, Eq. (6) of the paper."""
    k = sample_k(a.shape[0], a.device)
    eps = torch.randn_like(a)
    x = k[:, None] * eps + (1 - k[:, None]) * a
    return F.mse_loss(net(x, k, obs), eps - a)


def k_grid(n_steps: int = N_DENOISE):
    return [1.0 - i / n_steps for i in range(n_steps + 1)]


@torch.no_grad()
def euler_sample(vel_fn, obs, noise, n_steps: int = N_DENOISE):
    """Algorithm 2 of the paper with an arbitrary velocity field vel_fn(x, k, obs)."""
    x = noise
    ks = k_grid(n_steps)
    for k_cur, k_next in zip(ks[:-1], ks[1:]):
        k = torch.full((x.shape[0],), k_cur, device=x.device)
        x = x + (k_next - k_cur) * vel_fn(x, k, obs)
    return x


class EMA:
    def __init__(self, net: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(net).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, net: nn.Module, step: int):
        d = min(self.decay, (1 + step) / (10 + step))
        for s, p in zip(self.shadow.parameters(), net.parameters()):
            s.lerp_(p, 1 - d)


def save_ckpt(path, net: nn.Module, ema: EMA | None = None, **meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": net.config,
        "state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
        "ema_state_dict": {k: v.cpu() for k, v in ema.shadow.state_dict().items()} if ema else None,
        "meta": meta,
    }, path)


def load_ckpt(path, device, use_ema: bool = True) -> VelocityNet:
    ck = torch.load(path, map_location="cpu")
    net = VelocityNet(**ck["config"])
    sd = ck["ema_state_dict"] if use_ema and ck["ema_state_dict"] is not None else ck["state_dict"]
    net.load_state_dict(sd)
    return net.to(device).eval()
