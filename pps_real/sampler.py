"""Steering capable flow matching sampler for openpi's PI0Pytorch. Runs in the openpi venv.

sample_actions below is a line by line copy of PI0Pytorch.sample_actions at openpi commit 215abfb
(src/openpi/models_pytorch/pi0_pytorch.py) with two additions and no other change.

1. After the base velocity v_base is computed at each Euler step, an optional hook is called with
   (step, x_t, t, v_base, ctx). If it returns a tensor, that tensor replaces v_base in the Euler update.
   If it returns None, v_base is used unchanged. The hook never sees tensors it can mutate in a way that
   changes the base path: x_t, t and v_base are passed as they are, except t, which is a clone because
   the stock loop advances the time tensor in place.
2. The final layer prefix hidden states (the output of the PaliGemma backbone over image and language
   tokens, which the stock sampler computes and discards) are exposed through ctx.prefix_out, so proxies
   can read the base's frozen vision language features without a second forward pass.

The stock sampler is wrapped in torch.compile when the model config sets pytorch_compile_mode. This file
calls the model's own submodules (embed_prefix, paligemma_with_expert.forward, denoise_step) uncompiled,
so exact equality holds against the stock sampler with compilation disabled. See scripts/stage2/
check_sampler.py and results/stage2/sampler_check.json.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import torch
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks


@dataclasses.dataclass
class SamplerContext:
    """Per call information available to the hook. Built once per sample_actions call."""

    num_steps: int
    prefix_out: torch.Tensor  # (B, L, 2048) final layer PaliGemma hidden states over the prefix tokens
    prefix_pad_masks: torch.Tensor  # (B, L) True for valid prefix tokens
    state: torch.Tensor  # (B, 32) normalized state, unused by pi05_libero
    dt: float


# hook(step, x_t, t, v_base, ctx) -> new velocity or None
VelocityHook = Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor, SamplerContext], torch.Tensor | None]


@torch.no_grad()
def sample_actions(
    model: PI0Pytorch,
    device,
    observation,
    noise=None,
    num_steps: int = 10,
    hook: VelocityHook | None = None,
    prefix_callback: Callable[[SamplerContext], None] | None = None,
) -> torch.Tensor:
    """Same contract as PI0Pytorch.sample_actions, plus the velocity hook. Returns x_0 (B, H, D)."""
    bsize = observation.state.shape[0]
    if noise is None:
        actions_shape = (bsize, model.config.action_horizon, model.config.action_dim)
        noise = model.sample_noise(actions_shape, device)

    images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(observation, train=False)  # noqa: SLF001

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    # Compute image and language key value cache
    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)  # noqa: SLF001
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

    (prefix_out, _), past_key_values = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )

    ctx = SamplerContext(num_steps=num_steps, prefix_out=prefix_out, prefix_pad_masks=prefix_pad_masks,
                         state=state, dt=-1.0 / num_steps)
    if prefix_callback is not None:
        prefix_callback(ctx)

    dt = -1.0 / num_steps
    dt = torch.tensor(dt, dtype=torch.float32, device=device)

    x_t = noise
    time = torch.tensor(1.0, dtype=torch.float32, device=device)
    step = 0
    while time >= -dt / 2:
        expanded_time = time.expand(bsize)
        v_t = model.denoise_step(
            state,
            prefix_pad_masks,
            past_key_values,
            x_t,
            expanded_time,
        )
        if hook is not None:
            v_new = hook(step, x_t, expanded_time.clone(), v_t, ctx)
            if v_new is not None:
                v_t = v_new

        # Euler step - use new tensor assignment instead of in-place operation
        x_t = x_t + dt * v_t
        time += dt
        step += 1
    return x_t


class Recorder:
    """Hook that records (x_t, t, v_base) at every step and leaves the velocity unchanged."""

    def __init__(self):
        self.xs: list[torch.Tensor] = []
        self.ts: list[torch.Tensor] = []
        self.vs: list[torch.Tensor] = []
        self.ctx: SamplerContext | None = None

    def __call__(self, step, x_t, t, v_base, ctx):
        self.xs.append(x_t)
        self.ts.append(t)
        self.vs.append(v_base)
        self.ctx = ctx
        return None

    def stacked(self):
        """(K, B, H, D), (K, B), (K, B, H, D)."""
        return torch.stack(self.xs), torch.stack(self.ts), torch.stack(self.vs)


def pps_hook(v_task_fn, v_ref_fn, gamma: float) -> VelocityHook:
    """PPS, Eq. 4. v = v_base + gamma * (v_task - v_ref). Proxies take (x_t, t, ctx)."""

    def hook(step, x_t, t, v_base, ctx):
        if gamma == 0.0:
            return None
        return v_base + gamma * (v_task_fn(x_t, t, ctx) - v_ref_fn(x_t, t, ctx))

    return hook
