"""Shared optimisation loop. AdamW, linear warmup then cosine decay, EMA of weights."""

import math
import time

import torch

from toy.model import EMA


def train(net, batch_loss, steps: int, lr: float, warmup: int = 500, weight_decay: float = 1e-6,
          ema_decay: float = 0.999, log_every: int = 1000, desc: str = "") -> tuple[EMA, list]:
    """batch_loss(step) -> scalar loss tensor. Returns the EMA copy and a loss log."""
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))
    )
    ema = EMA(net, ema_decay)
    net.train()
    log, acc, t0 = [], 0.0, time.time()
    for step in range(steps):
        loss = batch_loss(step)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        ema.update(net, step)
        acc = acc + loss.detach()
        if (step + 1) % log_every == 0 or step + 1 == steps:
            n = log_every if (step + 1) % log_every == 0 else (step + 1) % log_every
            mean = float(acc) / n
            log.append({"step": step + 1, "loss": mean})
            print(f"[{desc}] step {step + 1}/{steps} loss {mean:.5f} ({time.time() - t0:.0f}s)", flush=True)
            acc = 0.0
    net.eval()
    return ema, log
