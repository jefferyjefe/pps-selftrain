"""The eval_reference.py checks, run on the paper style (image based) reference proxy. eval_reference.py is unchanged.

The proxy's image tokens are encoded once per frame (as at sampling time) and cached. An adapter takes frame indices
in place of the feature tensor the feature based proxy takes, so eval_reference.test2 and train_reference.evaluate
run unchanged on a GPUData whose .feats holds frame indices.

Test 1 (held out demos, teacher forced on the stored base path), with the same metrics (eval_reference.test1_metrics)
and the same seeded shuffle of held out frames as eval_reference.py.
  a. normal images
  b. both camera images shuffled across held out frames (each frame gets another frame's agentview and wrist pair)
  c. camera features set to zero, meaning the DINOv3 output tokens (the proxy's camera features) are zeroed, the
     learned camera embedding is still added
  c_black_images. extra: both input images all black, run through the encoder
Test 2 is eval_reference.test2 as is, on the held out demos and on train demos 0, 10, 20, 30, 40.

Usage
  /workspace/openpi/.venv/bin/python scripts/stage2/eval_reference_paper.py
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from eval_reference import TRAIN_DEMOS_T2, test1_metrics, test2  # noqa: E402
from paper_proxy import load_proxy  # noqa: E402
from train_reference import GPUData, evaluate  # noqa: E402
from train_reference_paper import load_images  # noqa: E402


class FrameAdapter:
    """Callable like ReferenceNet, net(frames, x_t, t), with frame indices in place of features."""

    def __init__(self, net, tokens, sub=256):
        self.net, self.tokens, self.sub = net, tokens, sub

    def eval(self):
        return self

    def train(self):
        return self

    def __call__(self, frames, x, t):
        out = []
        for i in range(0, len(frames), self.sub):
            kv = self.net.image_kv(self.tokens[frames[i:i + self.sub]])
            out.append(self.net.velocity(kv, x[i:i + self.sub], t[i:i + self.sub]))
        return torch.cat(out)


@torch.no_grad()
def encode_all(net, images, mode="normal", chunk=128):
    """Image tokens (F, 402, width) for every frame, in bf16 as under autocast at training time."""
    out = []
    for i in range(0, len(images), chunk):
        im = images[i:i + chunk]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if mode == "zero_features":
                N, C = im.shape[:2]
                tok = torch.zeros(N, C, 201, net.encoder.embed_dim, device=im.device) + net.cam_emb[None]
                out.append(net.img_proj(tok.flatten(1, 2)))
            elif mode == "black":
                out.append(net.encode(torch.zeros_like(im)))
            else:
                out.append(net.encode(im))
    return torch.cat(out)


@torch.no_grad()
def predict(adapter, data, ids, tokens_override=None, chunk=16384):
    if tokens_override is not None:
        adapter = FrameAdapter(adapter.net, tokens_override)
    out = []
    for i in range(0, len(ids), chunk):
        b = ids[i:i + chunk]
        _, x, t, _ = data.batch(b)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(adapter(b // data.per, x, t).float())
    return torch.cat(out)


@torch.no_grad()
def test1(adapter, net, images, data, tokens, metrics_ref, seed):
    ids = data.val_ids
    _, x, t, v_base = data.batch(ids)
    step = ids % data.steps
    val_frames = torch.unique(ids // data.per)
    g = torch.Generator(device="cpu").manual_seed(seed)  # same permutation as eval_reference.test1
    perm = val_frames[torch.randperm(len(val_frames), generator=g).to(val_frames.device)]
    src = torch.arange(len(tokens), device=val_frames.device)
    src[val_frames] = perm
    demo = torch.from_numpy(np.load(pathlib.Path(data.data_dir) / "demo_idx.npy")).to(perm.device)
    out = {"n_tuples": len(ids), "n_frames": len(val_frames),
           "shuffle": {"seed": seed, "what": "both camera images of each held out frame replaced by another held out "
                       "frame's pair", "frames_mapped_to_themselves": int((perm == val_frames).sum()),
                       "frames_mapped_within_same_demo": int((demo[perm] == demo[val_frames]).sum())}}
    ev = evaluate(adapter, data, ids)
    out["repro"] = {"metrics_json_best_val_mse_all32": metrics_ref, "train_reference_evaluate_mse_all32": ev["mse_all32"]}
    variants = {"a_normal": tokens, "b_shuffled_camera": tokens[src],
                "c_zero_camera": encode_all(net, images, "zero_features"),
                "c_black_images": encode_all(net, images, "black")}
    for name, tok in variants.items():
        out[name] = test1_metrics(predict(adapter, data, ids, tok), v_base, x, t, step, data.steps)
        if name == "a_normal":
            got = out[name]["velocity"]["mse_all32"]
            rel = abs(got - metrics_ref) / metrics_ref
            out["repro"].update(this_script_mse_all32=got, rel_diff=rel)
            print(f"repro: metrics.json {metrics_ref:.6f}, evaluate() {ev['mse_all32']:.6f}, here {got:.6f}", flush=True)
            if rel > 0.02:
                raise SystemExit(f"held out MSE does not reproduce (rel diff {rel:.3%}), stopping")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/workspace/data/t59_distill")
    p.add_argument("--images", default="/workspace/data/t59_demos/policy_obs.npz")
    p.add_argument("--ckpt", default="/workspace/runs/reference_t59_paper/best.pt")
    p.add_argument("--metrics", default="/workspace/runs/reference_t59_paper/metrics.json")
    p.add_argument("--out", default="/workspace/runs/reference_t59_paper/eval_checks.json")
    p.add_argument("--shuffle-seed", type=int, default=0)
    args = p.parse_args()
    d = pathlib.Path(args.data)
    man = json.loads((d / "manifest.json").read_text())
    seed_noise = int(man["noise"].split("((")[1].split(",")[0])
    dev = torch.device("cuda")
    data = GPUData(d, dev)
    data.data_dir = args.data
    images = load_images(args.images, data, dev)
    data.feats = torch.arange(len(images), device=dev)  # frame indices stand in for features
    torch.cuda.empty_cache()
    net = load_proxy(args.ckpt, dev)
    tokens = encode_all(net, images)
    adapter = FrameAdapter(net, tokens)
    metrics_ref = json.loads(pathlib.Path(args.metrics).read_text())["best_val"]["mse_all32"]

    res = {"ckpt": args.ckpt, "data": args.data, "images": args.images, "t_values": data.t.tolist(),
           "convention": "openpi 215abfb pi0_pytorch.py: x_t = t*noise + (1-t)*a, u = noise - a, so a_hat = x_t - t*v; "
                         "sampling x <- x + dt*v from t = 1.0, dt = -0.1 float32"}
    res["test1_heldout"] = test1(adapter, net, images, data, tokens, metrics_ref, args.shuffle_seed)
    print("test1 done", flush=True)
    demo = np.load(d / "demo_idx.npy")
    x0_all = np.load(d / "x0.npy").reshape(len(demo), data.draws, 10, 32)
    res["test2"] = {"noise_recipe": man["noise"] + ", frame is the global frame index 0..F-1"}
    for name, demos in {"heldout": data.val_demos, "train_demos_" + "_".join(map(str, TRAIN_DEMOS_T2)): TRAIN_DEMOS_T2}.items():
        frames = np.nonzero(np.isin(demo, demos))[0]
        res["test2"][name] = dict(demos=demos, **test2(adapter, data, x0_all, frames, seed_noise))
        r = res["test2"][name]
        print(f"test2 {name}: euler {r['euler_check']}, final mse7 {r['final_first7']['ref_vs_base_mse']:.5f} "
              f"vs draw pair {r['final_first7']['base_draw_pair_mse']:.5f}", flush=True)
    pathlib.Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
