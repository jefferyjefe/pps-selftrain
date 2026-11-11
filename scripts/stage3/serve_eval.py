"""Policy server for the stage 3 t59 evaluation. Runs in the openpi venv.

Same websocket protocol as scripts/stage2/serve.py: the client sends the flow matching noise under "__noise" (10, 32)
float32 and gets "actions" (10, 7) back. Inference is pps_real.policy.SteerablePolicy, i.e. pps_real.sampler eager
(no torch.compile), batch 1. One server process serves several clients; calls are handled one at a time, so every call
is a batch 1 call regardless of how many clients are connected.

Every server loads the base and both proxies and serves both methods. The client picks per request:
  "__method": "base"   pi0.5 alone, no hook (the stage 2 base).
  "__method": "pps"    PPS hook from pps_steer.py with "__gamma". The response also carries "pps", the per step cosines.
  "__method": "task"   the task proxy alone (pps_steer.py task_only_infer, or TaskOnlyHook with --task-impl hook).
                       Added for the 4090 rerun; base and pps are unchanged.
The response echoes the method and gamma. Calls do not share state (check_gamma0.py interleaves base and PPS calls in one
process and is bit exact), so which server process handles a call does not change its result.

Usage
  PYTHONPATH=. /workspace/openpi/.venv/bin/python scripts/stage3/serve_eval.py --port 8010
"""

import argparse
import json
import logging
import pathlib
import sys
import time

import numpy as np
import torch
from openpi.serving import websocket_policy_server

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pps_real import policy as _policy  # noqa: E402

REF = "/workspace/runs/reference_t59_paper_all50/final.pt"
TASK = "/workspace/runs/task_t59_paper_all50/final.pt"


class Server:
    def __init__(self, sp, steer, task_impl="fast"):
        self.sp = sp
        self.steer = steer
        self.task_impl = task_impl
        from pps_steer import TaskOnlyHook
        self.task_hook = TaskOnlyHook(steer)
        self.metadata = sp.metadata

    def infer(self, obs):
        noise = np.asarray(obs.pop("__noise"), dtype=np.float32)
        assert noise.shape == (10, 32), noise.shape
        method = str(obs.pop("__method"))
        gamma = float(obs.pop("__gamma", 0.0))
        assert method in ("base", "pps", "task"), method
        t0 = time.monotonic()
        hook = None
        if method == "task":
            if self.task_impl == "hook":
                self.task_hook.begin(np.asarray(obs["observation/image"]), np.asarray(obs["observation/wrist_image"]))
                out = self.sp.infer(obs, noise=noise, hook=self.task_hook)
            else:
                from pps_steer import task_only_infer
                out = task_only_infer(self.sp, self.steer, obs, noise)
            torch.cuda.synchronize()
            out["server_stats"] = {"infer_ms": (time.monotonic() - t0) * 1000, "method": method, "gamma": None,
                                   "task_impl": self.task_impl,
                                   "max_mem_allocated_gb": torch.cuda.max_memory_allocated() / 1e9}
            return out
        if method == "pps":
            self.steer.gamma = gamma
            self.steer.begin(np.asarray(obs["observation/image"]), np.asarray(obs["observation/wrist_image"]))
            hook = self.steer
        out = self.sp.infer(obs, noise=noise, hook=hook)
        torch.cuda.synchronize()
        out["server_stats"] = {"infer_ms": (time.monotonic() - t0) * 1000, "method": method,
                               "gamma": gamma if hook else None,
                               "max_mem_allocated_gb": torch.cuda.max_memory_allocated() / 1e9}
        if hook is not None:
            out["pps"] = self.steer.stats()
        return out

    def reset(self):
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", default=REF)
    p.add_argument("--task", default=TASK)
    p.add_argument("--checkpoint", default=_policy.DEFAULT_CHECKPOINT)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--task-impl", default="fast", choices=("fast", "hook"))
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    sp = _policy.SteerablePolicy(_policy.load_openpi_policy(args.checkpoint, compile_mode=None))
    from pps_steer import PPSSteer
    steer = PPSSteer(args.ref, args.task, 0.0)
    meta = dict(sp.metadata)
    meta["eval"] = {"port": args.port, "ref": args.ref, "task": args.task, "checkpoint": args.checkpoint,
                    "num_denoise_steps": sp.num_steps, "compile_mode": None, "gpu": torch.cuda.get_device_name(),
                    "task_impl": args.task_impl,
                    "sampler": "pps_real.sampler eager, batch 1; base: no hook; pps: scripts/stage3/pps_steer.py hook"}
    print(json.dumps(meta["eval"]), flush=True)
    websocket_policy_server.WebsocketPolicyServer(Server(sp, steer, args.task_impl), host="127.0.0.1", port=args.port,
                                                  metadata=meta).serve_forever()


if __name__ == "__main__":
    main()
