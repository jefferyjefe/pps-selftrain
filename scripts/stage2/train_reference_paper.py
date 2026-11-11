"""Train the paper style reference proxy for t59 (PPS arXiv 2609.09148v3, Eq. 5, Appendix A.1 and A.3).

Same distillation data as train_reference.py (x_t, t, v_base on the base's own 10 step path, 16 draws per frame),
but the observation is the two camera images pi0.5 saw (/workspace/data/t59_demos/policy_obs.npz, the file
collect_distill.py read, same frame order) instead of pi0.5's prefix features. Model in paper_proxy.py.

Recipe. 20,000 gradient steps at batch 64. One batch item is one (frame, draw) trajectory. Its two images are encoded
once and the expert is run at all 10 denoising steps, and the loss is the per step MSE summed over the 10 steps
(Eq. 5), averaged over the batch. No regularization (weight decay 0, no augmentation, no dropout, no EMA). AdamW,
bf16 autocast, linear warmup then cosine decay to 0, gradient clipping at 1.0 (openpi's default). Demos in the
manifest's split.val_demos are held out. Held out loss is logged every --eval-every steps and the best checkpoint
kept, and all steps are run regardless.

Usage
  /workspace/openpi/.venv/bin/python scripts/stage2/train_reference_paper.py --out /workspace/runs/reference_t59_paper
  smoke test: add --steps 300 --eval-every 150 --out <scratch dir>
  paper setting (all 50 demos, no held out set, final.pt is the model):
    ... --all-demos --out /workspace/runs/reference_t59_paper_all50
"""

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from paper_proxy import PaperProxy, n_params, save_proxy  # noqa: E402
from train_reference import GPUData  # noqa: E402


def load_images(path, data, dev):
    z = np.load(path)
    d = pathlib.Path(data.data_dir)
    assert np.array_equal(z["demo_idx"], np.load(d / "demo_idx.npy")), "frame order differs from the distillation data"
    assert np.array_equal(z["frame_idx"], np.load(d / "frame_idx.npy")), "frame order differs from the distillation data"
    img = torch.from_numpy(z["image"]).to(dev)
    wrist = torch.from_numpy(z["wrist_image"]).to(dev)
    return torch.stack([img, wrist], 1)  # (F, 2, 224, 224, 3) uint8, camera 0 agentview, camera 1 wrist


@torch.no_grad()
def evaluate(net, images, data, frames, chunk_frames=32):
    """All 16 draws x 10 steps of the given frames. Images encoded once per frame."""
    net.eval()
    D, S, dev = data.draws, data.steps, images.device
    se32 = torch.zeros(S, device=dev, dtype=torch.float64)
    se7, cs, n = torch.zeros_like(se32), torch.zeros_like(se32), torch.zeros_like(se32)
    for i in range(0, len(frames), chunk_frames):
        fr = frames[i:i + chunk_frames]
        tup = (fr[:, None] * data.per + torch.arange(data.per, device=dev)[None]).flatten()  # (frame, draw, step)
        x, v, t = data.x[tup], data.v[tup], data.t[tup % S]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = net.velocity(net.image_kv(net.encode(images[fr])), x, t, rep=data.per).float()
        step = tup % S
        e = (p - v) ** 2
        se32.index_add_(0, step, e.mean((1, 2)).double())
        se7.index_add_(0, step, e[..., :7].mean((1, 2)).double())
        cs.index_add_(0, step, torch.nn.functional.cosine_similarity(p.flatten(1), v.flatten(1), dim=1).double())
        n.index_add_(0, step, torch.ones_like(step, dtype=torch.float64))
    net.train()
    return {"loss_sum_over_steps": float((se32 / n).sum()), "mse_all32": float(se32.sum() / n.sum()),
            "mse_first7": float(se7.sum() / n.sum()), "cos": float(cs.sum() / n.sum()),
            "mse_all32_per_step": [float(z) for z in se32 / n], "cos_per_step": [float(z) for z in cs / n]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/workspace/data/t59_distill")
    p.add_argument("--images", default="/workspace/data/t59_demos/policy_obs.npz")
    p.add_argument("--out", default="/workspace/runs/reference_t59_paper")
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4, help="peak LR for the action expert and image projection")
    p.add_argument("--lr-encoder", type=float, default=2e-5, help="peak LR for the pretrained DINOv3 encoder")
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--all-demos", action="store_true",
                   help="train on all 50 demos (paper setting): no held out demos, no best.pt, only a train probe is "
                        "evaluated for logging, final.pt is the model")
    p.add_argument("--n-probe", type=int, default=505, help="train probe frames when --all-demos (505 = old val size)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    data = GPUData(pathlib.Path(args.data), dev)
    data.data_dir = args.data
    del data.feats  # the paper style proxy reads images, not pi0.5 features
    torch.cuda.empty_cache()
    images = load_images(args.images, data, dev)
    D = data.draws
    demo = torch.from_numpy(np.load(pathlib.Path(args.data) / "demo_idx.npy")).to(dev)
    if args.all_demos:
        data.val_demos = []
    is_val = torch.isin(demo, torch.tensor(data.val_demos, device=dev, dtype=demo.dtype))
    val_frames = torch.nonzero(is_val).squeeze(1)
    train_frames = torch.nonzero(~is_val).squeeze(1)
    train_traj = (train_frames[:, None] * D + torch.arange(D, device=dev)[None]).flatten()
    g = torch.Generator(device=dev).manual_seed(args.seed)
    n_probe = args.n_probe if args.all_demos else len(val_frames)
    probe_frames = train_frames[torch.randperm(len(train_frames), device=dev, generator=g)[:n_probe]].sort().values
    print(f"loaded in {time.time() - t0:.0f}s: {len(train_frames)} train frames ({len(train_traj)} trajectories), "
          f"{len(val_frames)} held out frames (demos {data.val_demos}), GPU mem {torch.cuda.memory_allocated() / 1e9:.1f} GB",
          flush=True)

    net = PaperProxy().to(dev)
    n_enc = n_params(net.encoder)
    print(f"PaperProxy trainable params {n_params(net):,} (encoder {n_enc:,}, expert {n_params(net) - n_enc:,})", flush=True)
    enc_ids = {id(q) for q in net.encoder.parameters()}
    groups = [{"params": list(net.encoder.parameters()), "lr": args.lr_encoder, "peak": args.lr_encoder},
              {"params": [q for q in net.parameters() if id(q) not in enc_ids], "lr": args.lr, "peak": args.lr}]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0, fused=True)

    def lr_scale(s):
        if s < args.warmup:
            return (s + 1) / args.warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - args.warmup) / max(1, args.steps - args.warmup))))

    config = dict(vars(args), model=net.config, encoder="timm " + "vit_small_patch16_dinov3.lvd1689m (pretrained)",
                  n_params=n_params(net), n_params_encoder=n_enc, n_train_trajectories=len(train_traj),
                  n_val_frames=len(val_frames), val_demos=data.val_demos, weight_decay=0.0, betas=[0.9, 0.95],
                  schedule="linear warmup then cosine to 0", loss="per step MSE over (10, 32), summed over 10 steps, "
                  "mean over batch", torch=torch.__version__, gpu=torch.cuda.get_device_name())
    (out / "config.json").write_text(json.dumps(config, indent=2))

    S, per = data.steps, data.per
    steps_ar = torch.arange(S, device=dev)
    perm, pos = train_traj[torch.randperm(len(train_traj), device=dev, generator=g)], 0
    history, best, run_loss, run_n = [], None, torch.zeros((), device=dev), 0
    t_start = time.time()
    for step in range(1, args.steps + 1):
        if pos + args.batch > len(perm):
            perm, pos = train_traj[torch.randperm(len(train_traj), device=dev, generator=g)], 0
        traj = perm[pos:pos + args.batch]
        pos += args.batch
        tup = (traj[:, None] * S + steps_ar[None]).flatten()  # (traj, step), traj = frame * D + draw
        x, v, t = data.x[tup], data.v[tup], data.t[tup % S]
        sc = lr_scale(step - 1)
        for gr in opt.param_groups:
            gr["lr"] = gr["peak"] * sc
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = net.velocity(net.image_kv(net.encode(images[traj // D])), x, t, rep=S)
        loss = ((pred.float() - v) ** 2).mean((1, 2)).view(args.batch, S).sum(1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
        opt.step()
        run_loss += loss.detach()
        run_n += 1
        if step % args.log_every == 0:
            el = time.time() - t_start
            print(f"  step {step} loss(sum over 10 steps) {float(run_loss / run_n):.4f} grad_norm {float(gn):.2f} "
                  f"lr {args.lr * sc:.2e} {el / step * 1000:.0f} ms/step elapsed {el / 60:.1f} min "
                  f"eta {el / step * (args.steps - step) / 60:.1f} min", flush=True)
            run_loss, run_n = torch.zeros((), device=dev), 0
        if step % args.eval_every == 0 or step == args.steps:
            te = time.time()
            tr = evaluate(net, images, data, probe_frames)
            if args.all_demos:
                rec = dict(step=step, train_probe={k: tr[k] for k in ("loss_sum_over_steps", "mse_all32", "mse_first7", "cos")},
                           elapsed_s=time.time() - t_start, eval_s=time.time() - te)
                history.append(rec)
                print(f"eval step {step}: train probe loss {tr['loss_sum_over_steps']:.4f} mse32 {tr['mse_all32']:.5f} "
                      f"mse7 {tr['mse_first7']:.5f} cos {tr['cos']:.4f} ({rec['eval_s']:.0f}s)", flush=True)
                (out / "metrics.json").write_text(json.dumps(dict(history=history), indent=2))
                continue
            ev = evaluate(net, images, data, val_frames)
            rec = dict(step=step, val=ev, train_probe={k: tr[k] for k in ("loss_sum_over_steps", "mse_all32", "mse_first7", "cos")},
                       elapsed_s=time.time() - t_start, eval_s=time.time() - te)
            history.append(rec)
            is_best = best is None or ev["mse_all32"] < best["val"]["mse_all32"]
            if is_best:
                best = dict(step=step, val=ev)
                save_proxy(out / "best.pt", net, step=step, val=ev, data=args.data, images=args.images,
                           val_demos=data.val_demos)
            print(f"eval step {step}: val loss {ev['loss_sum_over_steps']:.4f} mse32 {ev['mse_all32']:.5f} "
                  f"mse7 {ev['mse_first7']:.5f} cos {ev['cos']:.4f} | train probe mse32 {tr['mse_all32']:.5f} "
                  f"cos {tr['cos']:.4f} ({rec['eval_s']:.0f}s){' *' if is_best else ''}", flush=True)
            (out / "metrics.json").write_text(json.dumps(dict(best_step=best["step"], best_val=best["val"],
                                                              history=history), indent=2))
    save_proxy(out / "final.pt", net, step=args.steps, data=args.data, images=args.images, val_demos=data.val_demos)
    train_s = time.time() - t_start
    m = json.loads((out / "metrics.json").read_text())
    m["train_s"] = train_s
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    if args.all_demos:
        print(f"done in {train_s / 60:.1f} min, final.pt is the model (no held out set)", flush=True)
        return
    print(f"done in {train_s / 60:.1f} min, best step {best['step']} val mse32 {best['val']['mse_all32']:.5f} "
          f"cos {best['val']['cos']:.4f}", flush=True)


if __name__ == "__main__":
    main()
