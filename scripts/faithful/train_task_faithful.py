"""Faithful task proxy for t59: the official PPS task stage (github.com/TritiumR/pps @ a22f9ad, scripts/train_pytorch.py
with their ProxyPytorch.forward) on the 50 t59 demos.

Theirs, unchanged
  - Model and loss: their ProxyPytorch.forward(observation, actions) called as is: their augmentation
    (preprocess_observation_pytorch train=True, one draw per batch) and DINOv3 RoPE rescale in train mode, t ~
    Beta(1.5, 1) * 0.999 + 0.001, fresh N(0, I) noise, x_t = t * noise + (1 - t) * a, u = noise - a, per element MSE
    over (B, 10, 7) real dims, then .mean() (train_pytorch.py).
  - Initialization: every weight from the reference proxy (their --pytorch_weight_path, safetensors.load_model of the
    whole model), fresh optimizer, schedule restarted at step 0.
  - Optimizer and schedule: AdamW all parameters, betas (0.9, 0.95), eps 1e-8, weight decay 1e-10, clip 1.0, their
    lr_schedule (1000 warmup from peak/1001, cosine from 2.5e-5 to 2.5e-6 at 30,000). Batch 64 and 24,000 steps: the
    metadata.pt of their released pot and weight task checkpoints (task/24000, num_train_steps 24001, batch_size 64,
    save_interval 8000). Their tea task proxy used 32,000; pot and weight (2 of the 3 released tasks, and the README's
    Stage 2 example) used 24,000. fp32 + TF32. Seed 42 + seed index.
  - Data: frames sampled uniformly without replacement per epoch (their DataLoader shuffle). Action targets are the
    pi05_libero pipeline's normalized chunks (/workspace/data/t59_task/actions.npy, LeRobot chunking with the last
    action repeated past an episode end, as their loader does), real dims only.

Usage
  /workspace/openpi/.venv/bin/python scripts/faithful/train_task_faithful.py \
      --init /workspace/runs/reference_t59_faithful_s0/final.pt --out /workspace/runs/task_t59_faithful_s0
"""

import argparse
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import official  # noqa: E402
from distill_faithful import DATA, Frames, their_lr_schedule  # noqa: E402

TASK = pathlib.Path("/workspace/data/t59_task")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0, help="seed index; torch/numpy seed = 42 + seed (theirs is 42)")
    p.add_argument("--steps", type=int, default=24_000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--save-every", type=int, default=8000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--probe-every", type=int, default=1000)
    args = p.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    dev = torch.device("cuda")
    seed = 42 + args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    info = json.loads((DATA / "real_dims.json").read_text())
    A = info["action_dim_real"]
    net, cfg, init_meta = official.load_proxy(args.init, dev)
    assert cfg.action_dim == A
    torch.set_float32_matmul_precision("high")
    frames = Frames(dev, cfg)
    del frames.npz
    actions = torch.from_numpy(np.load(TASK / "actions.npy")[..., :A].copy()).to(dev)  # (F, 10, 7)
    assert len(actions) == frames.n

    # fixed probe (monitoring only): 256 frames, fixed noise and t, eval mode (no augmentation)
    g = torch.Generator(device=dev).manual_seed(4321)
    probe_idx = torch.arange(0, frames.n, frames.n // 256, device=dev)[:256]
    probe_noise = torch.randn((len(probe_idx), cfg.action_horizon, A), generator=g, device=dev)
    probe_t = torch.distributions.Beta(torch.tensor(1.5, device=dev), torch.tensor(1.0, device=dev)).sample(
        (len(probe_idx),)) * 0.999 + 0.001

    @torch.no_grad()
    def probe():
        net.eval()
        tot = []
        for i in range(0, len(probe_idx), 64):
            sl = slice(i, i + 64)
            obs = frames.proxy(probe_idx[sl])
            a = actions[probe_idx[sl]]
            t = probe_t[sl]
            x_t = t[:, None, None] * probe_noise[sl] + (1 - t[:, None, None]) * a
            u = probe_noise[sl] - a
            images, img_masks, state = net._preprocess_observation(obs, train=False)  # noqa: SLF001
            pe, pm, _ = net.embed_prefix(images, img_masks)
            se, sm, _, ad = net.embed_suffix(state, x_t, t)
            pad = torch.cat([pm, sm], 1)
            h, _ = net.expert_model.forward(attention_mask=pad, position_ids=(torch.cumsum(pad, 1) - 1).long(),
                                            past_key_values=None, inputs_embeds=torch.cat([pe, se], 1),
                                            use_cache=False, adarms_cond=ad)
            v = net.action_out_proj(h[:, -cfg.action_horizon:].float())
            tot.append(F.mse_loss(v, u, reduction="none").mean((1, 2)))
        net.train()
        return float(torch.cat(tot).mean())

    optim = torch.optim.AdamW(net.parameters(), lr=2.5e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10)
    commit = subprocess.run(["git", "-C", str(pathlib.Path(__file__).resolve().parents[2]), "rev-parse", "--short",
                             "HEAD"], capture_output=True, text=True).stdout.strip()
    config = dict(vars(args), torch_seed=seed, proxy_cfg=cfg.__dict__, init_meta=init_meta,
                  n_params=sum(q.numel() for q in net.parameters()), gpu=torch.cuda.get_device_name(),
                  torch=torch.__version__, repo_commit=commit, official_commit=official.COMMIT,
                  official_sources=official.SOURCES, float32_matmul_precision="high",
                  optimizer=dict(name="AdamW", betas=[0.9, 0.95], eps=1e-8, weight_decay=1e-10, clip=1.0),
                  lr_schedule=dict(warmup=1000, peak=2.5e-5, decay_steps=30000, end=2.5e-6),
                  released_metadata="pot/weight task/24000: batch_size 64, num_train_steps 24001, save_interval 8000, "
                                    "pytorch_weight_path = reference/20000",
                  data=dict(frames=frames.n, actions=str(TASK / "actions.npy") + f"[..., :{A}]"))
    (out / "config.json").write_text(json.dumps(config, indent=1, default=str))

    hist = {"train": [], "probe": [dict(step=0, mse=probe())]}
    print(f"probe step 0: {hist['probe'][-1]['mse']:.5f}", flush=True)
    rng = np.random.default_rng(seed)
    perm, pos = torch.from_numpy(rng.permutation(frames.n)).to(dev), 0
    net.train()
    t_start = time.time()
    run = []
    for step in range(args.steps):
        if pos + args.batch > frames.n:
            perm, pos = torch.from_numpy(rng.permutation(frames.n)).to(dev), 0
        idx = perm[pos:pos + args.batch]
        pos += args.batch
        for pg in optim.param_groups:
            pg["lr"] = their_lr_schedule(step)
        losses = net(frames.proxy(idx), actions[idx])
        loss = losses.mean()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)
        run.append((loss.detach(), gn.detach()))
        s1 = step + 1
        if s1 % args.log_every == 0:
            el = time.time() - t_start
            ls = float(torch.stack([a for a, _ in run]).mean())
            gs = float(torch.stack([b for _, b in run]).mean())
            run = []
            hist["train"].append(dict(step=s1, loss=ls, grad_norm=gs, lr=their_lr_schedule(step), elapsed_s=el))
            print(f"step {s1} loss {ls:.5f} grad_norm {gs:.3f} lr {their_lr_schedule(step):.2e} "
                  f"{el / s1 * 1000:.0f} ms/step eta {el / s1 * (args.steps - s1) / 60:.1f} min "
                  f"mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
        if s1 % args.probe_every == 0 or s1 == args.steps:
            hist["probe"].append(dict(step=s1, mse=probe(), elapsed_s=time.time() - t_start))
            print(f"probe step {s1}: {hist['probe'][-1]['mse']:.5f}", flush=True)
            (out / "metrics.json").write_text(json.dumps(hist, indent=1))
        if s1 % args.save_every == 0 or s1 == args.steps:
            name = "final.pt" if s1 == args.steps else f"step_{s1}.pt"
            official.save_proxy(out / name, net, cfg, step=s1, kind="task", seed=args.seed, init=args.init,
                                gpu=torch.cuda.get_device_name())
            print(f"saved {name}", flush=True)
    hist["train_s"] = time.time() - t_start
    hist["gpu"] = torch.cuda.get_device_name()
    (out / "metrics.json").write_text(json.dumps(hist, indent=1))
    print(f"done in {hist['train_s'] / 60:.1f} min on {hist['gpu']}", flush=True)


if __name__ == "__main__":
    main()
