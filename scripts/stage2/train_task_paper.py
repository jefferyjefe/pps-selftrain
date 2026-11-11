"""Train the paper style task proxy for t59 (PPS arXiv 2609.09148v3, Sec. 4.3 "Task specialization", Eq. 6, App. A.3).

Initialized from the reference proxy (Algorithm 1: v_task <- v_ref) and fine-tuned on the 50 t59 demos with openpi's
standard flow matching loss. No language input, encoder trained jointly. All 50 demos, 20,000 steps at batch 64,
no validation based stopping, final.pt is the model.

Data. /workspace/data/t59_task/actions.npy (build_task_data.py): the 5571 demo frames' action chunks after openpi's
own pi05_libero pipeline (LeRobot chunking, checkpoint quantile norm stats, padding to 32). Images are the frames
pi0.5 saw (/workspace/data/t59_demos/policy_obs.npz), the same input the reference proxy reads. One batch item is one
frame, sampled uniformly without replacement per epoch.

Loss. openpi PI0Pytorch.forward / sample_noise / sample_time, copied (see flow_matching_loss): t ~ Beta(1.5, 1)
* 0.999 + 0.001, noise ~ N(0, I) over all 32 dims, x_t = t * noise + (1 - t) * a, u_t = noise - a, per element MSE
over (B, 10, 32) and then .mean() as openpi's train_pytorch.py does, so all 32 dims (incl. the zero padded 25) count.

Optimizer. Same as the reference run (train_reference_paper.py): AdamW betas (0.9, 0.95), weight decay 0, peak LR 2e-4
for the expert and image projection, 2e-5 for the DINOv3 encoder, 1000 step linear warmup then cosine to 0, grad
clip 1.0, bf16 autocast. The paper does not give these.

Usage
  /workspace/openpi/.venv/bin/python scripts/stage2/train_task_paper.py \
      --init /workspace/runs/reference_t59_paper_all50/final.pt --out /workspace/runs/task_t59_paper_all50
  smoke test (low memory): add --steps 300 --images-on-cpu --mem-fraction 0.3 --out <scratch dir>
"""

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from paper_proxy import load_proxy, n_params, save_proxy  # noqa: E402


# ---- copied from openpi src/openpi/models_pytorch/pi0_pytorch.py (commit 215abfb) ----
def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def sample_noise(shape, device):
    return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)


def sample_time(bsize, device):
    time_beta = sample_beta(1.5, 1.0, bsize, device)
    time = time_beta * 0.999 + 0.001
    return time.to(dtype=torch.float32, device=device)


def flow_matching_loss(net, images, actions, noise=None, time=None):
    """PI0Pytorch.forward with the VLM + expert replaced by the proxy. Returns per element losses (B, 10, 32)."""
    if noise is None:
        noise = sample_noise(actions.shape, actions.device)
    if time is None:
        time = sample_time(actions.shape[0], actions.device)
    time_expanded = time[:, None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions
    with torch.autocast("cuda", dtype=torch.bfloat16):
        v_t = net(images, x_t, time)
    v_t = v_t.to(dtype=torch.float32)
    return F.mse_loss(u_t, v_t, reduction="none")
# ---- end copy (train_pytorch.py then takes losses.mean()) ----


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", default="/workspace/runs/reference_t59_paper_all50/final.pt")
    p.add_argument("--task-data", default="/workspace/data/t59_task")
    p.add_argument("--images", default="/workspace/data/t59_demos/policy_obs.npz")
    p.add_argument("--out", default="/workspace/runs/task_t59_paper_all50")
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr-encoder", type=float, default=2e-5)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--n-probe", type=int, default=512, help="frames in the fixed noise/time train probe")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--images-on-cpu", action="store_true", help="keep images in pinned CPU memory (low GPU memory)")
    p.add_argument("--mem-fraction", type=float, default=None, help="cap this process's share of GPU memory")
    args = p.parse_args()

    dev = torch.device("cuda")
    if args.mem_fraction:
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    td = pathlib.Path(args.task_data)
    t0 = time.time()
    z = np.load(args.images)
    assert np.array_equal(z["demo_idx"], np.load(td / "demo_idx.npy"))
    assert np.array_equal(z["frame_idx"], np.load(td / "frame_idx.npy"))
    images = torch.stack([torch.from_numpy(z["image"]), torch.from_numpy(z["wrist_image"])], 1)  # (F, 2, 224, 224, 3)
    images = images.pin_memory() if args.images_on_cpu else images.to(dev)
    actions = torch.from_numpy(np.load(td / "actions.npy")).to(dev)  # (F, 10, 32) normalized, padded
    Fn = len(actions)
    print(f"loaded {Fn} frames in {time.time() - t0:.0f}s, images on {'cpu' if args.images_on_cpu else 'gpu'}, "
          f"GPU mem {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)

    def get_images(idx):
        if args.images_on_cpu:
            return images[idx.cpu()].to(dev, non_blocking=True)
        return images[idx]

    net = load_proxy(args.init, dev)
    net.train()
    for q in net.parameters():
        q.requires_grad_(True)
    n_enc = n_params(net.encoder)
    print(f"init from {args.init}, trainable params {n_params(net):,} (encoder {n_enc:,})", flush=True)
    enc_ids = {id(q) for q in net.encoder.parameters()}
    groups = [{"params": list(net.encoder.parameters()), "lr": args.lr_encoder, "peak": args.lr_encoder},
              {"params": [q for q in net.parameters() if id(q) not in enc_ids], "lr": args.lr, "peak": args.lr}]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0, fused=True)

    def lr_scale(s):
        if s < args.warmup:
            return (s + 1) / args.warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - args.warmup) / max(1, args.steps - args.warmup))))

    # fixed probe: frames, noise and times drawn once, so the logged probe loss is comparable across steps
    g = torch.Generator(device=dev).manual_seed(args.seed + 1)
    probe = torch.randperm(Fn, device=dev, generator=g)[:args.n_probe].sort().values
    probe_noise = torch.randn((len(probe), 10, 32), device=dev, generator=g)
    probe_time = (torch.distributions.Beta(torch.tensor(1.5, device=dev), torch.tensor(1.0, device=dev))
                  .sample((len(probe),)) * 0.999 + 0.001)

    @torch.no_grad()
    def probe_loss():
        net.eval()
        tot, tot7 = 0.0, 0.0
        for i in range(0, len(probe), 64):
            idx = probe[i:i + 64]
            le = flow_matching_loss(net, get_images(idx), actions[idx], probe_noise[i:i + 64], probe_time[i:i + 64])
            tot += float(le.sum())
            tot7 += float(le[..., :7].sum())
        net.train()
        return tot / (len(probe) * 10 * 32), tot7 / (len(probe) * 10 * 7)

    config = dict(vars(args), model=net.config, n_params=n_params(net), n_params_encoder=n_enc, n_frames=Fn,
                  weight_decay=0.0, betas=[0.9, 0.95], schedule="linear warmup then cosine to 0",
                  loss="openpi PI0Pytorch flow matching: t ~ Beta(1.5,1)*0.999+0.001, x_t = t*noise + (1-t)*a, "
                       "u = noise - a, MSE mean over (B, 10, 32)",
                  sampling="frames uniform without replacement per epoch", torch=torch.__version__,
                  gpu=torch.cuda.get_device_name())
    (out / "config.json").write_text(json.dumps(config, indent=2))

    pl, pl7 = probe_loss()
    history = [dict(step=0, probe_loss=pl, probe_loss_first7=pl7)]
    print(f"eval step 0: probe loss {pl:.4f} (first 7 dims {pl7:.4f})", flush=True)
    gperm = torch.Generator(device=dev).manual_seed(args.seed)
    perm, pos = torch.randperm(Fn, device=dev, generator=gperm), 0
    run_loss, run_n, train_curve = torch.zeros((), device=dev), 0, []
    t_start = time.time()
    for step in range(1, args.steps + 1):
        if pos + args.batch > Fn:
            perm, pos = torch.randperm(Fn, device=dev, generator=gperm), 0
        idx = perm[pos:pos + args.batch]
        pos += args.batch
        sc = lr_scale(step - 1)
        for gr in opt.param_groups:
            gr["lr"] = gr["peak"] * sc
        loss = flow_matching_loss(net, get_images(idx), actions[idx]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
        opt.step()
        run_loss += loss.detach()
        run_n += 1
        if step % args.log_every == 0:
            el = time.time() - t_start
            avg = float(run_loss / run_n)
            train_curve.append(dict(step=step, loss=avg, grad_norm=float(gn), lr=args.lr * sc))
            print(f"  step {step} loss {avg:.4f} grad_norm {float(gn):.2f} lr {args.lr * sc:.2e} "
                  f"{el / step * 1000:.0f} ms/step elapsed {el / 60:.1f} min eta {el / step * (args.steps - step) / 60:.1f} min "
                  f"mem {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)
            run_loss, run_n = torch.zeros((), device=dev), 0
        if step % args.eval_every == 0 or step == args.steps:
            pl, pl7 = probe_loss()
            history.append(dict(step=step, probe_loss=pl, probe_loss_first7=pl7, elapsed_s=time.time() - t_start))
            print(f"eval step {step}: probe loss {pl:.4f} (first 7 dims {pl7:.4f})", flush=True)
            (out / "metrics.json").write_text(json.dumps(dict(probe=history, train=train_curve), indent=2))
    save_proxy(out / "final.pt", net, step=args.steps, init=args.init, task_data=args.task_data, images=args.images)
    train_s = time.time() - t_start
    (out / "metrics.json").write_text(json.dumps(dict(probe=history, train=train_curve, train_s=train_s), indent=2))
    print(f"done in {train_s / 60:.1f} min, final.pt is the model", flush=True)


if __name__ == "__main__":
    main()
