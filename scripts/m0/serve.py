"""Policy server for milestone 0 LIBERO evals. Runs inside the openpi venv.

Same websocket protocol as openpi's scripts/serve_policy.py, with three additions.
1. The client may send flow matching noise under the key "__noise" so every episode is reproducible.
2. Every response carries server stats (peak GPU memory, denoising steps, inference time).
3. Optional norm stats override, used to run pi05_base through the pi05_libero config.

Usage
  /workspace/openpi/.venv/bin/python scripts/m0/serve.py --checkpoint /workspace/checkpoints/pi05_libero_pytorch --port 8000
"""

import argparse
import dataclasses
import inspect
import json
import logging
import time

import numpy as np
import torch
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.shared import normalize
from openpi.training import config as _config


class NoisePolicy:
    """Wraps an openpi Policy so the client can pass the initial noise and read GPU stats."""

    def __init__(self, policy, num_steps):
        self._policy = policy
        self._num_steps = num_steps
        self.metadata = policy.metadata

    def infer(self, obs):
        noise = obs.pop("__noise", None)
        if noise is not None:
            noise = np.asarray(noise, dtype=np.float32)
        t0 = time.monotonic()
        out = self._policy.infer(obs, noise=noise)
        torch.cuda.synchronize()
        out["server_stats"] = {
            "infer_ms": (time.monotonic() - t0) * 1000,
            "max_mem_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "max_mem_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
            "num_denoise_steps": self._num_steps,
        }
        return out

    def reset(self):
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="pi05_libero")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--norm-stats", default=None, help="norm_stats.json to use instead of the checkpoint assets")
    p.add_argument("--compile-mode", default="default", help="'default' keeps the config value, 'none' disables torch.compile")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)

    cfg = _config.get_config(args.config)
    if args.compile_mode != "default":
        mode = None if args.compile_mode == "none" else args.compile_mode
        cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=mode))
    norm_stats = normalize.load(args.norm_stats.rsplit("/", 1)[0]) if args.norm_stats else None

    policy = policy_config.create_trained_policy(cfg, args.checkpoint, norm_stats=norm_stats)
    # Default number of denoising steps, read from the sampler signature rather than assumed.
    num_steps = inspect.signature(type(policy._model).sample_actions).parameters["num_steps"].default
    num_steps = policy._sample_kwargs.get("num_steps", num_steps)
    logging.info("config=%s checkpoint=%s denoise_steps=%s compile=%s", args.config, args.checkpoint, num_steps,
                 cfg.model.pytorch_compile_mode)
    wrapped = NoisePolicy(policy, num_steps)
    meta = dict(policy.metadata)
    meta["m0"] = {"config": args.config, "checkpoint": args.checkpoint, "num_denoise_steps": num_steps,
                  "norm_stats": args.norm_stats, "compile_mode": cfg.model.pytorch_compile_mode,
                  "action_horizon": cfg.model.action_horizon, "action_dim": cfg.model.action_dim}
    print(json.dumps(meta["m0"]), flush=True)
    websocket_policy_server.WebsocketPolicyServer(wrapped, host="0.0.0.0", port=args.port, metadata=meta).serve_forever()


if __name__ == "__main__":
    main()
