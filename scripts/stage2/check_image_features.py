"""Check that an image source is what pi0.5 saw during distillation collection. Runs in the openpi venv.

1. If --images is not the policy_obs.npz the collection read, pixel comparison against it (byte equality).
2. Prefix features. For a few groups of 4 consecutive frames (the collector's batches, frames 4j..4j+3, every
   frame repeated for its 16 draws with the same seeded noise, 64 rows), the base sampler is run on the re-rendered
   images exactly as collect_distill.py ran it, the draw 0 prefix is pooled with its pool_features, and the result is
   compared with features.npy. The batch is rebuilt exactly because bf16 kernels give batch size dependent results
   (docs/stage2.md part 2: features at batch 1 differ from the stored ones by up to 1 percent per token).

Usage
  /workspace/openpi/.venv/bin/python scripts/stage2/check_image_features.py
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

from pps_real import policy as _policy
from pps_real import sampler as _sampler

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from check_sampler import batch_observation  # noqa: E402
from collect_distill import N_IMG_TOK, pool_features  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", default="/workspace/data/t59_demos/policy_obs.npz")
    p.add_argument("--orig-obs", default="/workspace/data/t59_demos/policy_obs.npz")
    p.add_argument("--distill", default="/workspace/data/t59_distill")
    p.add_argument("--groups", default="0,300,700,1100,1390", help="batch indices j, frames 4j..4j+3")
    p.add_argument("--out", default="/workspace/runs/reference_t59_paper/feature_check.json")
    args = p.parse_args()
    d = pathlib.Path(args.distill)
    man = json.loads((d / "manifest.json").read_text())
    seed, D = int(man["noise"].split("((")[1].split(",")[0]), man["draws_per_frame"]
    z = np.load(args.images)
    image, wrist, state, prompt = z["image"], z["wrist_image"], z["state"], str(z["prompt"])
    assert prompt == man["prompt"], prompt
    pix = "skipped, --images is the collector's own input"
    if pathlib.Path(args.images).resolve() != pathlib.Path(args.orig_obs).resolve():
      zo = np.load(args.orig_obs)
      pix = {k: dict(equal=bool(np.array_equal(z[k], zo[k])), max_abs_diff=int(np.abs(z[k].astype(np.int16) - zo[k]).max()))
           for k in ("image", "wrist_image")}
      pix["state_equal"] = bool(np.array_equal(z["state"], zo["state"]))
    print("pixels vs original policy_obs.npz:", pix, flush=True)

    feats = np.load(d / "features.npy", mmap_mode="r")
    pol = _policy.load_openpi_policy(compile_mode=None)
    sp = _policy.SteerablePolicy(pol)
    model, dev = sp.model, sp.device
    rows = []
    for j in map(int, args.groups.split(",")):
        fr = list(range(4 * j, 4 * j + 4))
        obs = []
        for f in fr:
            o = {"observation/image": image[f], "observation/wrist_image": wrist[f],
                 "observation/state": state[f].astype(np.float64), "prompt": prompt}
            obs += [o] * D
        ob = batch_observation(sp, obs)
        noise = np.stack([np.random.default_rng((seed, f, dd)).standard_normal((10, 32)).astype(np.float32)
                          for f in fr for dd in range(D)])
        rec = _sampler.Recorder()
        with torch.no_grad():
            _sampler.sample_actions(model, dev, ob, noise=torch.from_numpy(noise).to(dev), hook=rec)
        ctx = rec.ctx
        rows0 = torch.arange(len(fr), device=dev) * D
        n_lang = ctx.prefix_out.shape[1] - 3 * N_IMG_TOK
        fe = pool_features(ctx.prefix_out[rows0], ctx.prefix_pad_masks[rows0], n_lang).cpu().numpy().astype(np.float16)
        for i, f in enumerate(fr):
            ref = np.asarray(feats[f])
            a, b = fe[i].astype(np.float32), ref.astype(np.float32)
            cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
            rows.append(dict(frame=f, bit_equal_fp16=bool(np.array_equal(fe[i], ref)),
                             max_abs_diff=float(np.abs(a - b).max()), min_token_cos=float(cos.min())))
            print(rows[-1], flush=True)
    res = dict(pixels_vs_original=pix, frames=rows,
               all_features_bit_equal=all(r["bit_equal_fp16"] for r in rows),
               max_abs_diff=max(r["max_abs_diff"] for r in rows), min_token_cos=min(r["min_token_cos"] for r in rows))
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "frames"}, indent=1))


if __name__ == "__main__":
    main()
