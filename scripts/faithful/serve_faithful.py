"""Policy server for the faithful-proxy t59 evaluation. openpi venv. Same websocket protocol as scripts/stage3/serve_eval.py
(client sends "__noise" (10, 32) float32, "__method", "__gamma"; batch 1; calls handled one at a time).

Methods
  base      pi0.5 alone: SteerablePolicy.infer with no hook (the stock eager sampler; identical to serve_eval.py's base).
  pps_f     faithful PPS (faithful_steer.FaithfulSteer mode "pps", real dims only) at "__gamma".
  blend_f   v_real = (1 - gamma) v_base + gamma v_task, real dims only, faithful task proxy, at "__gamma".
  task_f    faithful task proxy alone = their --only_steer (fast path by default; --task-impl hook runs the base).
  pps_v3    v3 ours PPS (scripts/stage3/pps_steer.PPSSteer, all 32 dims, bf16 autocast), unchanged from Nov 11.
  task_v3   v3 ours task proxy alone (scripts/stage3/pps_steer.task_only_infer, all 32 dims), unchanged from Nov 11.
The response carries per-step velocity cosines for the steered methods and, for the faithful hook methods, the running
count of padding-dim checks and violations (v[..., 7:] must equal v_base[..., 7:] bit for bit at every Euler step).

Usage
  PYTHONPATH=. /workspace/openpi/.venv/bin/python scripts/faithful/serve_faithful.py --port 8020
"""

import argparse
import json
import logging
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "stage3"))
import official  # noqa: E402,F401
from faithful_steer import FaithfulSteer, task_only_fast  # noqa: E402
from openpi.serving import websocket_policy_server  # noqa: E402

from pps_real import policy as _policy  # noqa: E402

REF_F = "/workspace/runs/reference_t59_faithful_s0/final.pt"
TASK_F = "/workspace/runs/task_t59_faithful_s0/final.pt"
REF_V3 = "/workspace/runs/reference_t59_paper_all50/final.pt"
TASK_V3 = "/workspace/runs/task_t59_paper_all50/final.pt"
MODES = {"pps_f": "pps", "blend_f": "blend", "task_f": "only"}


class Server:
    def __init__(self, sp, steer, v3, task_impl="fast"):
        self.sp, self.steer, self.v3, self.task_impl = sp, steer, v3, task_impl
        self.metadata = sp.metadata

    def infer(self, obs):
        noise = np.asarray(obs.pop("__noise"), dtype=np.float32)
        assert noise.shape == (10, 32), noise.shape
        method = str(obs.pop("__method"))
        gamma = float(obs.pop("__gamma", 0.0))
        t0 = time.monotonic()
        extra = {}
        if method == "base":
            out = self.sp.infer(obs, noise=noise)
        elif method == "task_f" and self.task_impl == "fast":
            out = task_only_fast(self.sp, self.steer, obs, noise)
        elif method in MODES:
            inputs, _ = self.sp.observation(obs)
            self.steer.begin(np.asarray(obs["observation/image"]), np.asarray(obs["observation/wrist_image"]),
                             inputs["state"][0], MODES[method], gamma if method != "task_f" else 1.0)
            out = self.sp.infer(obs, noise=noise, hook=self.steer)
            out["pps"] = self.steer.stats()
            extra = {"pad_checked": self.steer.pad_checked, "pad_violations": self.steer.pad_violations}
        elif method in ("pps_v3", "task_v3") and self.v3 is None:
            raise ValueError("server started with --no-v3")
        elif method == "pps_v3":
            self.v3.gamma = gamma
            self.v3.begin(np.asarray(obs["observation/image"]), np.asarray(obs["observation/wrist_image"]))
            out = self.sp.infer(obs, noise=noise, hook=self.v3)
            out["pps"] = self.v3.stats()
        elif method == "task_v3":
            from pps_steer import task_only_infer
            out = task_only_infer(self.sp, self.v3, obs, noise)
        else:
            raise ValueError(method)
        torch.cuda.synchronize()
        out["server_stats"] = {"infer_ms": (time.monotonic() - t0) * 1000, "method": method,
                               "gamma": gamma if method in ("pps_f", "blend_f", "pps_v3") else None,
                               "max_mem_allocated_gb": torch.cuda.max_memory_allocated() / 1e9, **extra}
        return out

    def reset(self):
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--ref", default=REF_F)
    p.add_argument("--task", default=TASK_F)
    p.add_argument("--ref-v3", default=REF_V3)
    p.add_argument("--task-v3", default=TASK_V3)
    p.add_argument("--checkpoint", default=_policy.DEFAULT_CHECKPOINT)
    p.add_argument("--task-impl", default="fast", choices=("fast", "hook"))
    p.add_argument("--no-faithful", action="store_true", help="serve base and v3 only (before the faithful proxies exist)")
    p.add_argument("--no-v3", action="store_true", help="do not load the t59 v3 proxies (stage 4: other tasks)")
    p.add_argument("--task-id", type=int, default=59, help="libero_90 task the faithful proxies belong to (metadata)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    sp = _policy.SteerablePolicy(_policy.load_openpi_policy(args.checkpoint, compile_mode=None))
    prec = torch.get_float32_matmul_precision()  # "high": set by upstream PI0Pytorch.__init__ (pi0_pytorch.py:111)
    assert prec == "high", prec
    steer = None if args.no_faithful else FaithfulSteer(args.ref, args.task)
    v3 = None
    if not args.no_v3:
        from pps_steer import PPSSteer
        v3 = PPSSteer(args.ref_v3, args.task_v3, 0.0)
    assert torch.get_float32_matmul_precision() == prec
    meta = dict(sp.metadata)
    meta["eval"] = {"port": args.port, "ref": None if args.no_faithful else args.ref,
                    "task": None if args.no_faithful else args.task,
                    "task_id": None if args.no_faithful else args.task_id,
                    "ref_v3": None if args.no_v3 else args.ref_v3,
                    "task_v3": None if args.no_v3 else args.task_v3, "checkpoint": args.checkpoint,
                    "num_denoise_steps": sp.num_steps,
                    "compile_mode": None, "gpu": torch.cuda.get_device_name(), "task_impl": args.task_impl,
                    "official_commit": official.COMMIT,
                    "sampler": "pps_real.sampler eager, batch 1; base: no hook; *_f: scripts/faithful/faithful_steer.py; "
                               "*_v3: scripts/stage3/pps_steer.py"}
    print(json.dumps(meta["eval"]), flush=True)
    websocket_policy_server.WebsocketPolicyServer(Server(sp, steer, v3, args.task_impl), host="127.0.0.1",
                                                  port=args.port, metadata=meta).serve_forever()


if __name__ == "__main__":
    main()
