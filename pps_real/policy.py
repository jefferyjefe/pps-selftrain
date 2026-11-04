"""openpi Policy wrapper that samples with pps_real.sampler. Runs in the openpi venv.

SteerablePolicy.infer is a copy of openpi Policy.infer (PyTorch branch, openpi commit 215abfb) that calls
pps_real.sampler.sample_actions instead of model.sample_actions and forwards a velocity hook.
"""

from __future__ import annotations

import dataclasses
import time

import jax
import numpy as np
import torch
from openpi.models import model as _model
from openpi.policies import policy as _policy
from openpi.policies import policy_config
from openpi.training import config as _config

from pps_real import sampler as _sampler

DEFAULT_CHECKPOINT = "/workspace/checkpoints/pi05_libero_pytorch"


def load_openpi_policy(checkpoint: str = DEFAULT_CHECKPOINT, config: str = "pi05_libero",
                       compile_mode: str | None = None) -> _policy.Policy:
    """Stock openpi policy. compile_mode None disables torch.compile, 'default' keeps the config value."""
    cfg = _config.get_config(config)
    if compile_mode != "default":
        cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=compile_mode))
    return policy_config.create_trained_policy(cfg, checkpoint)


class SteerablePolicy:
    def __init__(self, policy: _policy.Policy):
        assert policy._is_pytorch_model  # noqa: SLF001
        self.policy = policy
        self.model = policy._model  # noqa: SLF001
        self.device = policy._pytorch_device  # noqa: SLF001
        self.num_steps = policy._sample_kwargs.get("num_steps", 10)  # noqa: SLF001
        self.metadata = policy.metadata

    def observation(self, obs: dict) -> tuple[dict, _model.Observation]:
        """Input transforms and batching exactly as Policy.infer. Returns (inputs, Observation) with batch 1."""
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self.policy._input_transform(inputs)  # noqa: SLF001
        inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self.device)[None, ...], inputs)
        return inputs, _model.Observation.from_dict(inputs)

    def infer(self, obs: dict, *, noise: np.ndarray | None = None, hook=None, prefix_callback=None) -> dict:
        inputs, observation = self.observation(obs)
        sample_kwargs = dict(self.policy._sample_kwargs)  # noqa: SLF001
        if noise is not None:
            noise = torch.from_numpy(noise).to(self.device)
            if noise.ndim == 2:
                noise = noise[None, ...]
            sample_kwargs["noise"] = noise
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": _sampler.sample_actions(self.model, self.device, observation, hook=hook,
                                               prefix_callback=prefix_callback, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        outputs = self.policy._output_transform(outputs)  # noqa: SLF001
        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        return outputs

    def postprocess(self, state: torch.Tensor, x0: torch.Tensor) -> np.ndarray:
        """Output transform (unnormalize, slice to 7 dims) for one raw model output, as Policy.infer does."""
        out = {"state": np.asarray(state.detach().cpu()), "actions": np.asarray(x0.detach().cpu())}
        return self.policy._output_transform(out)["actions"]  # noqa: SLF001
