"""Stage 4 memory fix check: are the reference-training batches bit identical with the prefix cache in pinned host
memory (the t59 path) and in plain host memory (the stage 4 path for tasks whose pinned cache would not fit)?

Runs the setup of scripts/faithful/distill_faithful.py main() up to the prefix cache (same seeds, same teacher, same
frames, same build_prefix_cache), with the pin choice forced by PPS_FAITHFUL_PIN_LIMIT (0: plain; large: pinned),
then the batch assembly of its training loop (rng = default_rng(42 + seed), permutation, GROUP batches of
--batch frames, K[flat] / V[flat], .to(cuda, non_blocking=True)) for the first --groups groups. Writes sha256 of
the whole K and V caches and, per group, of the frame ids, the host batch (K[flat], V[flat]) and the device batch
(copied back to host). Run once per path and compare the two json files with --compare.

Usage (openpi venv, repo root, PPS_FAITHFUL_* set for the task):
  PPS_FAITHFUL_PIN_LIMIT=0 python scripts/multitask/check_pin_batches.py --out plain.json
  PPS_FAITHFUL_PIN_LIMIT=1000000000000 python scripts/multitask/check_pin_batches.py --out pinned.json
  python scripts/multitask/check_pin_batches.py --compare plain.json pinned.json --out compare.json
"""

import argparse
import hashlib
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "faithful"))


def sha(t):
    import torch
    t = t.contiguous()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


def sha_big(t, chunk=256):
    h = hashlib.sha256()
    for i in range(0, t.shape[0], chunk):
        import torch
        c = t[i:i + chunk].contiguous()
        if c.dtype == torch.bfloat16:
            c = c.view(torch.int16)
        h.update(c.numpy().tobytes())
    return h.hexdigest()


def run(a):
    import numpy as np
    import torch
    import distill_faithful as D
    from pps_real import policy as _policy

    dev = torch.device("cuda")
    seed = 42 + a.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    info = json.loads((D.DATA / "real_dims.json").read_text())
    cfg = D.official.ProxyCfg(action_dim=info["action_dim_real"], state_dim=info["state_dim_real"],
                              action_horizon=info["action_horizon"])
    t0 = time.time()
    sp = _policy.SteerablePolicy(_policy.load_openpi_policy(compile_mode=None))
    model = sp.model
    model.eval()
    for q in model.parameters():
        q.requires_grad = False
    frames = D.Frames(dev, cfg)
    del frames.npz
    torch.set_float32_matmul_precision("high")
    K, V, _ = D.build_prefix_cache(model, frames)
    print(f"cache built at {time.time() - t0:.0f}s, pinned={K.is_pinned()}", flush=True)
    out = {"pin_limit": D.PIN_LIMIT_BYTES, "pinned": bool(K.is_pinned()), "frames": int(frames.n),
           "shape": list(K.shape), "gpu": torch.cuda.get_device_name(), "data": str(D.DATA), "obs": D.OBS,
           "K_sha256": sha_big(K), "V_sha256": sha_big(V), "groups": []}
    print("cache hashed", flush=True)
    rng = np.random.default_rng(seed)
    perm, pos = rng.permutation(frames.n), 0
    for g in range(a.groups):
        ids = []
        for _ in range(D.GROUP):
            if pos + a.batch > frames.n:
                perm, pos = rng.permutation(frames.n), 0
            ids.append(perm[pos:pos + a.batch])
            pos += a.batch
        flat = torch.from_numpy(np.concatenate(ids))
        kc, vc = K[flat], V[flat]
        kd, vd = kc.to(dev, non_blocking=True), vc.to(dev, non_blocking=True)
        torch.cuda.synchronize()
        out["groups"].append({"ids": sha(flat), "k_host": sha(kc), "v_host": sha(vc),
                              "k_dev": sha(kd.cpu()), "v_dev": sha(vd.cpu())})
        if g % 100 == 0:
            print(f"group {g} {time.time() - t0:.0f}s", flush=True)
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out, flush=True)


def compare(a):
    x, y = (json.loads(pathlib.Path(p).read_text()) for p in a.compare)
    n = min(len(x["groups"]), len(y["groups"]))
    diff = [g for g in range(n) if x["groups"][g] != y["groups"][g]]
    res = {"files": a.compare, "pinned": [x["pinned"], y["pinned"]], "frames": [x["frames"], y["frames"]],
           "gpu": [x["gpu"], y["gpu"]], "K_identical": x["K_sha256"] == y["K_sha256"],
           "V_identical": x["V_sha256"] == y["V_sha256"], "groups_compared": n,
           "steps_compared": n * 4, "groups_differing": diff[:20], "n_groups_differing": len(diff)}
    res["IDENTICAL_ALL"] = (res["K_identical"] and res["V_identical"] and not diff and n > 0
                            and x["pinned"] != y["pinned"])
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--groups", type=int, default=500)
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    (compare if a.compare else run)(a)


if __name__ == "__main__":
    main()
