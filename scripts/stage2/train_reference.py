"""Train the reference proxy for t59 by distilling the frozen pi0.5 base velocities (paper Eq. 5, Appendix A.3).

Data is the on policy dataset written by collect_distill.py (/workspace/data/t59_distill). One example is
(features[frame], x_t[frame, draw, step], t[step]) with target v_base[frame, draw, step]. Demos in the manifest's
split.val_demos are held out. The whole dataset is put on the GPU once and batches are drawn by index.

Model. The toy VelocityNet layout (toy/model.py), reusing its sinusoidal time embedding and conditioned ResBlock.
The noised (10, 32) chunk is concatenated with the time embedding and fed to the residual trunk, and the observation
conditions every block. The observation is the base's 33 pooled prefix tokens (2048 wide). Each token is projected
to a small width with a shared linear layer and a LayerNorm, the 33 tokens are flattened, and an MLP maps them to
the trunk width. Time is openpi's t (1 = noise), embedded with toy.model.sinusoidal (which scales by 1000).

Training. MSE on all 32 dims, AdamW, bf16 autocast, cosine LR with a short linear warmup, held out eval every
epoch, best checkpoint kept, early stopping on held out MSE, hard wall clock cap.

Usage
  /workspace/openpi/.venv/bin/python scripts/stage2/train_reference.py --out /workspace/runs/reference_t59
  smoke test: add --max-minutes 1 --out <scratch dir>
"""

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from toy.model import ResBlock, n_params, sinusoidal  # noqa: E402

N_TOK, FEAT_DIM, H, D = 33, 2048, 10, 32


class ReferenceNet(nn.Module):
    def __init__(self, width=512, depth=3, tok_dim=64, temb_dim=128, n_tok=N_TOK, feat_dim=FEAT_DIM,
                 horizon=H, act_dim=D):
        super().__init__()
        self.config = dict(width=width, depth=depth, tok_dim=tok_dim, temb_dim=temb_dim, n_tok=n_tok,
                           feat_dim=feat_dim, horizon=horizon, act_dim=act_dim)
        self.temb_dim, self.horizon, self.act_dim = temb_dim, horizon, act_dim
        self.tok_proj = nn.Sequential(nn.Linear(feat_dim, tok_dim), nn.LayerNorm(tok_dim))
        self.obs_in = nn.Sequential(nn.Linear(n_tok * tok_dim, width), nn.GELU(), nn.Linear(width, width))
        self.act_in = nn.Sequential(nn.Linear(horizon * act_dim + temb_dim, width), nn.GELU(),
                                    nn.Linear(width, width))
        self.blocks = nn.ModuleList([ResBlock(width) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, horizon * act_dim))

    def forward(self, feats, x_t, t):
        """feats (B, 33, 2048), x_t (B, 10, 32), t (B,) openpi time. Returns v (B, 10, 32)."""
        B = x_t.shape[0]
        c = self.obs_in(self.tok_proj(feats).flatten(1))
        h = self.act_in(torch.cat([x_t.reshape(B, -1), sinusoidal(t, self.temb_dim).to(x_t.dtype)], dim=-1))
        for blk in self.blocks:
            h = blk(h, c)
        return self.out(h).view(B, self.horizon, self.act_dim)


def load_reference(path, device="cuda"):
    ck = torch.load(path, map_location="cpu")
    net = ReferenceNet(**ck["config"])
    net.load_state_dict(ck["state_dict"])
    return net.to(device).eval()


class GPUData:
    """All arrays on the GPU. Tuple id = (frame * draws + draw) * steps + step."""

    def __init__(self, d, dev):
        man = json.loads((d / "manifest.json").read_text())
        t0 = time.time()
        self.feats = torch.from_numpy(np.load(d / "features.npy")).to(dev)  # (F, 33, 2048) fp16
        x = np.load(d / "x_t.npy")
        Fn, self.draws, self.steps = x.shape[:3]
        self.x = torch.from_numpy(x).to(dev).view(-1, H, D)
        del x
        self.v = torch.from_numpy(np.load(d / "v_base.npy")).to(dev).view(-1, H, D)
        self.t = torch.from_numpy(np.load(d / "t.npy")).to(dev)
        demo = np.load(d / "demo_idx.npy")
        self.val_demos = man["split"]["val_demos"]
        is_val = torch.from_numpy(np.isin(demo, self.val_demos)).to(dev)
        per = self.draws * self.steps
        frame_of = torch.arange(Fn * per, device=dev) // per
        self.train_ids = torch.nonzero(~is_val[frame_of]).squeeze(1)
        self.val_ids = torch.nonzero(is_val[frame_of]).squeeze(1)
        self.per = per
        self.load_s = time.time() - t0

    def batch(self, ids):
        return self.feats[ids // self.per], self.x[ids], self.t[ids % self.steps], self.v[ids]


@torch.no_grad()
def evaluate(net, data, ids, chunk=16384):
    """MSE (all 32, first 7), per step, and cosine of v_ref vs v_base over the flattened (10, 32) chunk."""
    net.eval()
    S = data.steps
    se32, se7 = torch.zeros(S, device=ids.device, dtype=torch.float64), torch.zeros(S, device=ids.device,
                                                                                    dtype=torch.float64)
    cos_sum, cos7_sum = torch.zeros_like(se32), torch.zeros_like(se32)
    cnt = torch.zeros_like(se32)
    for i in range(0, len(ids), chunk):
        b = ids[i:i + chunk]
        f, x, t, v = data.batch(b)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = net(f, x, t)
        p = p.float()
        step = b % S
        e = (p - v) ** 2
        se32.index_add_(0, step, e.mean(dim=(1, 2)).double())
        se7.index_add_(0, step, e[..., :7].mean(dim=(1, 2)).double())
        cos_sum.index_add_(0, step, F.cosine_similarity(p.flatten(1), v.flatten(1), dim=1).double())
        cos7_sum.index_add_(0, step, F.cosine_similarity(p[..., :7].flatten(1), v[..., :7].flatten(1), dim=1).double())
        cnt.index_add_(0, step, torch.ones_like(step, dtype=torch.float64))
    net.train()
    tot = cnt.sum()
    r = lambda a: [float(z) for z in (a / cnt)]  # noqa: E731
    return {
        "n_tuples": int(tot),
        "mse_all32": float(se32.sum() / tot), "mse_first7": float(se7.sum() / tot),
        "cos": float(cos_sum.sum() / tot), "cos_first7": float(cos7_sum.sum() / tot),
        "mse_all32_per_step": r(se32), "mse_first7_per_step": r(se7),
        "cos_per_step": r(cos_sum), "cos_first7_per_step": r(cos7_sum),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/workspace/data/t59_distill")
    p.add_argument("--out", default="/workspace/runs/reference_t59")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--tok-dim", type=int, default=64)
    p.add_argument("--temb-dim", type=int, default=128)
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=300)
    p.add_argument("--epochs", type=int, default=60, help="length of the cosine schedule")
    p.add_argument("--patience", type=int, default=8, help="epochs without held out improvement before stopping")
    p.add_argument("--min-delta", type=float, default=0.005, help="relative improvement that counts")
    p.add_argument("--max-minutes", type=float, default=20.0, help="hard cap on training wall clock")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = GPUData(pathlib.Path(args.data), dev)
    print(f"loaded data to GPU in {data.load_s:.1f}s: {len(data.train_ids):,} train tuples, "
          f"{len(data.val_ids):,} held out tuples (demos {data.val_demos}), "
          f"GPU mem {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
    g = torch.Generator(device=dev).manual_seed(args.seed)
    train_probe = data.train_ids[torch.randperm(len(data.train_ids), device=dev, generator=g)[:len(data.val_ids)]]

    net = ReferenceNet(args.width, args.depth, args.tok_dim, args.temb_dim).to(dev)
    print(f"ReferenceNet params {n_params(net):,}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95), fused=True)
    steps_per_epoch = len(data.train_ids) // args.batch
    total_steps = args.epochs * steps_per_epoch

    def lr_at(s):
        if s < args.warmup_steps:
            return args.lr * (s + 1) / args.warmup_steps
        q = min(1.0, (s - args.warmup_steps) / max(1, total_steps - args.warmup_steps))
        return args.lr * 0.5 * (1 + math.cos(math.pi * q))

    config = dict(vars(args), model=net.config, n_params=n_params(net), steps_per_epoch=steps_per_epoch,
                  total_steps_planned=total_steps, n_train_tuples=len(data.train_ids),
                  n_val_tuples=len(data.val_ids), val_demos=data.val_demos,
                  torch=torch.__version__, gpu=torch.cuda.get_device_name())
    (out / "config.json").write_text(json.dumps(config, indent=2))

    history, best, bad_epochs, step, stop_reason = [], None, 0, 0, "schedule finished"
    t_start = time.time()
    cap_s = args.max_minutes * 60
    for epoch in range(args.epochs):
        perm = data.train_ids[torch.randperm(len(data.train_ids), device=dev, generator=g)]
        run_loss, run_n, t_ep = torch.zeros((), device=dev), 0, time.time()
        for i in range(steps_per_epoch):
            for pg in opt.param_groups:
                pg["lr"] = lr_at(step)
            f, x, t, v = data.batch(perm[i * args.batch:(i + 1) * args.batch])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = net(f, x, t)
            loss = F.mse_loss(pred.float(), v)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_loss += loss.detach()
            run_n += 1
            step += 1
            if step % args.log_every == 0:
                print(f"  step {step} lr {lr_at(step):.2e} loss {float(run_loss / run_n):.5f} "
                      f"elapsed {time.time() - t_start:.0f}s", flush=True)
            if time.time() - t_start > cap_s:
                stop_reason = f"wall clock cap {args.max_minutes} min"
                break
        if stop_reason.startswith("wall") and run_n < steps_per_epoch // 2 and epoch > 0:
            break  # partial epoch too short to be worth an eval
        ev = evaluate(net, data, data.val_ids)
        tr = evaluate(net, data, train_probe)
        rec = dict(epoch=epoch + 1, step=step, lr=lr_at(step), train_loss_running=float(run_loss / max(run_n, 1)),
                   train_probe_mse_all32=tr["mse_all32"], val_mse_all32=ev["mse_all32"],
                   val_mse_first7=ev["mse_first7"], val_cos=ev["cos"], epoch_s=time.time() - t_ep,
                   elapsed_s=time.time() - t_start)
        history.append(rec)
        improved = best is None or ev["mse_all32"] < best["val"]["mse_all32"] * (1 - args.min_delta)
        if best is None or ev["mse_all32"] < best["val"]["mse_all32"]:
            best = dict(epoch=epoch + 1, step=step, val=ev)
            torch.save({"config": net.config, "state_dict": {k: w.detach().cpu() for k, w in net.state_dict().items()},
                        "meta": dict(epoch=epoch + 1, step=step, val_mse_all32=ev["mse_all32"],
                                     data=args.data, val_demos=data.val_demos)}, out / "best.pt")
        bad_epochs = 0 if improved else bad_epochs + 1
        print(f"epoch {epoch + 1} step {step} train_run {rec['train_loss_running']:.5f} "
              f"train_probe {tr['mse_all32']:.5f} val32 {ev['mse_all32']:.5f} val7 {ev['mse_first7']:.5f} "
              f"cos {ev['cos']:.4f} ({rec['epoch_s']:.1f}s){' *' if best['epoch'] == epoch + 1 else ''}",
              flush=True)
        if stop_reason.startswith("wall"):
            break
        if bad_epochs >= args.patience:
            stop_reason = f"early stop, no {args.min_delta:.1%} held out improvement for {args.patience} epochs"
            break
    train_s = time.time() - t_start
    print(f"stopped: {stop_reason}, {train_s / 60:.1f} min, best epoch {best['epoch']}", flush=True)

    net = load_reference(out / "best.pt", dev)
    val = evaluate(net, data, data.val_ids)
    train = evaluate(net, data, data.train_ids)
    baselines_path = pathlib.Path(args.data) / "baselines.json"
    baselines = json.loads(baselines_path.read_text()) if baselines_path.exists() else None
    metrics = dict(best_epoch=best["epoch"], best_step=best["step"], stop_reason=stop_reason, train_s=train_s,
                   t_values=data.t.tolist(), val=val, train=train,
                   baselines=None if baselines is None else {
                       k: {m: baselines[k][m] for m in ("mse_all32", "mse_first7")}
                       for k in ("a_mean_per_dim", "b_mean_per_step_horizon_dim")},
                   history=history)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"best ckpt: val mse32 {val['mse_all32']:.5f} mse7 {val['mse_first7']:.5f} cos {val['cos']:.4f} | "
          f"train mse32 {train['mse_all32']:.5f} cos {train['cos']:.4f}", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
