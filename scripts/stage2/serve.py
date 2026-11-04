"""Policy server for stage 2. Runs in the openpi venv.

Same websocket protocol as scripts/m0/serve.py (the client sends the flow matching noise under "__noise" and every
response carries server stats), but inference goes through pps_real.policy.SteerablePolicy, the eager sampler that
was verified bit exact against openpi's stock sampler. With no hook this is the frozen base. Proxies and PPS
hooks plug in here in later stages.

Usage
  PYTHONPATH=. /workspace/openpi/.venv/bin/python scripts/stage2/serve.py --port 8002
"""

import argparse
import json
import logging
import time

import numpy as np
import torch
from openpi.serving import websocket_policy_server

from pps_real import policy as _policy


class Server:
    def __init__(self, sp: _policy.SteerablePolicy, hook=None):
        self.sp = sp
        self.hook = hook
        self.metadata = sp.metadata

    def infer(self, obs):
        noise = obs.pop("__noise", None)
        if noise is not None:
            noise = np.asarray(noise, dtype=np.float32)
        t0 = time.monotonic()
        out = self.sp.infer(obs, noise=noise, hook=self.hook)
        torch.cuda.synchronize()
        out["server_stats"] = {
            "infer_ms": (time.monotonic() - t0) * 1000,
            "max_mem_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "max_mem_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
            "num_denoise_steps": self.sp.num_steps,
        }
        return out

    def reset(self):
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=_policy.DEFAULT_CHECKPOINT)
    p.add_argument("--port", type=int, default=8002)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    sp = _policy.SteerablePolicy(_policy.load_openpi_policy(args.checkpoint, compile_mode=None))
    meta = dict(sp.metadata)
    meta["m0"] = {"config": "pi05_libero", "checkpoint": args.checkpoint, "num_denoise_steps": sp.num_steps,
                  "norm_stats": None, "compile_mode": None, "sampler": "pps_real.sampler eager, no hook"}
    print(json.dumps(meta["m0"]), flush=True)
    websocket_policy_server.WebsocketPolicyServer(Server(sp), host="0.0.0.0", port=args.port,
                                                  metadata=meta).serve_forever()


if __name__ == "__main__":
    main()
