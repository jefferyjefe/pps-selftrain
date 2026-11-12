"""Faithful reference proxy for t59: the official PPS distillation (github.com/TritiumR/pps @ a22f9ad) on pi05_libero.

What is theirs, unchanged
  - Model: their ProxyPytorch (DINOv3 ViT-S/16 via Meta's dinov3 code + gemma_12m HF Gemma expert over
    [196 + 196 image tokens, 1 state token, 10 action tokens], HF's causal mask from the 2-D pad mask, RoPE positions,
    pi0 sin/cos time embedding, fp32), loaded through scripts/faithful/official.py. LIBERO sizes: 7 real action dims,
    8 real state dims, both read from the pi05_libero norm stats.
  - Augmentation: their preprocess_observation_pytorch(train=True) inside the student's forward (exterior camera crop
    to 95% + resize back + rotation in [-5, 5] deg; brightness 0.7-1.3, contrast 0.6-1.4, saturation 0.5-1.5 on both
    cameras; one draw per batch), plus Meta DINOv3's RoPE coordinate rescale (log-uniform x1/2..x2) in train mode.
    The teacher sees clean images (their PI0Pytorch.forward_for_distill preprocesses with train=False).
  - Distillation targets: their PI0Pytorch.forward_for_distill, online every step: fresh N(0, I) noise over all 32
    dims, 10 stratified random times (one uniform draw in each 0.1 bin, descending), the teacher integrates its own
    Euler path 1.0 -> tau_1 -> ... -> tau_10 -> 0 and records (x_t, t, v_base) at each of the 11 points. Their
    forward_distill drops the t = 1.0 point and keeps the real dims, so the loss is mean over (B, 10, 10, 7) of
    (v_student - v_base)^2, on x_t taken from the teacher's path (use_noise_for_distill=True).
  - Optimizer and schedule: AdamW over all parameters, betas (0.9, 0.95), eps 1e-8, weight decay 1e-10, grad clip 1.0,
    LR peak 2.5e-5 with 1000 linear warmup steps from peak/1001, cosine to 2.5e-6 at decay_steps 30,000 (their
    lr_schedule function, copied). Batch 8 and 20,000 steps: the values recorded in the metadata.pt of their released
    reference checkpoints (pot and weight; reference/20000, num_train_steps 20001, batch_size 8). fp32 weights with TF32
    matmuls (float32_matmul_precision "high", set by their ProxyPytorch.__init__). Seed 42 + seed index.

What is restructured (same function, verified on a batch before training starts; results in config.json)
  - Teacher prefix. pi05_libero ignores state and the prompt is the same for every frame, so the teacher's PaliGemma
    prefix key/value cache depends only on the frame. It is computed once per frame (exactly their prefix code) and
    kept on the host, valid tokens only (2 x 256 image + 13 prompt tokens; the masked third image slot and the padded
    prompt positions are dropped, which only removes keys whose softmax weight is exactly 0). Each step then runs
    their forward_for_distill from the noise draw on, with that cache and an all-valid pad mask. Teacher calls are
    made for 4 training steps at once (32 frames), which changes nothing since the teacher is frozen.
    Check (a): labels vs their uncached forward_for_distill with the same RNG state.
  - Student. Their forward_distill repeats the 392 image tokens for each of the 10 path points and runs 10 x 403
    tokens through the expert. Under the causal mask the image tokens never attend to the state or action tokens, so
    running the image tokens once per observation and the 11 state + action tokens per path point against the
    cached image keys and values is the same computation. Check (b): loss and every gradient vs their forward_distill
    with the same RNG state (same augmentation draw).

Inputs: the 5571 frames of the 50 t59 demos (/workspace/data/t59_demos/policy_obs.npz, the 224 frames pi0.5 sees),
normalized state from /workspace/data/t59_faithful (build_faithful_data.py), the pi05_libero pytorch checkpoint as
teacher (pps_real.policy.load_openpi_policy, eager).

Usage
  /workspace/openpi/.venv/bin/python scripts/faithful/distill_faithful.py --out /workspace/runs/reference_t59_faithful_s0
"""

import argparse
import concurrent.futures
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import official  # noqa: E402  (sets OPENPI_DISABLE_TORCH_COMPILE before openpi imports)
from openpi.models import model as _model  # noqa: E402
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402

from pps_real import policy as _policy  # noqa: E402

# Stage 4: other tasks point these at their own inputs (scripts/multitask/build_task_inputs.sh); defaults are t59's.
DATA = pathlib.Path(os.environ.get("PPS_FAITHFUL_DATA", "/workspace/data/t59_faithful"))
OBS = os.environ.get("PPS_FAITHFUL_OBS", "/workspace/data/t59_demos/policy_obs.npz")
GROUP = 4  # training steps per teacher call


# ---- copied from their openpi/src/openpi/models_pytorch/pi0_pytorch.py (a22f9ad) ----
def sample_bin(start, end, bsize, num_samples, device):
    bin_width = (end - start) / num_samples
    offsets = (
        torch.arange(num_samples, dtype=torch.float32, device=device) * bin_width
        + start
    )
    offsets = offsets.unsqueeze(0)
    noise = (
        torch.rand(bsize, num_samples, dtype=torch.float32, device=device) * bin_width
    )
    return offsets + noise


def sample_bin_times(bsize, num_steps, device):
    time = sample_bin(0.0, 1.0, bsize, num_steps, device)
    time = time.flip(dims=[1])
    return time.to(dtype=torch.float32, device=device)


@torch.no_grad()
def their_forward_for_distill(self, observation, num_steps, teacher_flow_path_noise_std: float = 0.0):
    """PI0Pytorch.forward_for_distill (their pi0_pytorch.py:439-560), verbatim with `self` = our PI0Pytorch.
    Used only for check (a)."""
    images, img_masks, lang_tokens, lang_masks, state = (
        self._preprocess_observation(observation, train=False)
    )
    state = torch.nn.functional.pad(
        state,
        (0, self.config.action_dim - state.shape[1]),
        mode="constant",
        value=0,
    )
    bsize = state.shape[0]
    device = state.device
    actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
    initial_noise = self.sample_noise(actions_shape, device)
    time_schedule = sample_bin_times(bsize, num_steps, device)
    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
        images, img_masks, lang_tokens, lang_masks
    )
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
    self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (
        "eager"  # noqa: SLF001
    )
    _, past_key_values = self.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    return _their_flow_path(self, state, prefix_pad_masks, past_key_values, initial_noise, time_schedule, bsize, device,
                            teacher_flow_path_noise_std)


def _their_flow_path(self, state, prefix_pad_masks, past_key_values, initial_noise, time_schedule, bsize, device,
                     teacher_flow_path_noise_std=0.0):
    """The rest of their forward_for_distill, from "Initialize for denoising process" on, verbatim."""
    x_t = initial_noise
    current_time = torch.tensor(1.0, dtype=torch.float32, device=device).expand(
        bsize
    )
    end_time = torch.tensor(0.0, dtype=torch.float32, device=device).expand(bsize)
    time_schedule = torch.cat([time_schedule, end_time.unsqueeze(1)], dim=1)
    time_schedule = time_schedule.transpose(0, 1)  # (num_steps+1, batch_size)
    noises = []
    times = []
    gradients = []
    for target_time in time_schedule:
        if teacher_flow_path_noise_std > 0:
            x_t.add_(
                torch.randn_like(x_t) * teacher_flow_path_noise_std
            )
        noises.append(x_t.clone())
        times.append(current_time.clone())
        v_t = self.denoise_step(
            state,
            prefix_pad_masks,
            past_key_values,
            x_t,
            current_time,
        )
        gradients.append(v_t.clone())
        dt = (target_time - current_time)[:, None, None]
        x_t = x_t + dt * v_t
        current_time = target_time
    noises = torch.stack(noises, dim=1)
    times = torch.stack(times, dim=1)
    gradients = torch.stack(gradients, dim=1)
    return noises, times, gradients, x_t


def their_lr_schedule(step: int, warmup_steps=1000, peak_lr=2.5e-5, decay_steps=30_000, end_lr=2.5e-6):
    """lr_schedule from their scripts/distill_pytorch.py and train_pytorch.py (identical in both)."""
    if step < warmup_steps:
        # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
        init_lr = peak_lr / (warmup_steps + 1)
        return init_lr + (peak_lr - init_lr) * step / warmup_steps
    # cosine decay
    progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
    cos = 0.5 * (1 + np.cos(np.pi * progress))
    return end_lr + (peak_lr - end_lr) * cos
# ---- end copy ----


class Frames:
    """The demo frames as openpi Observations, built on the GPU from the pipeline's arrays."""

    def __init__(self, dev, cfg):
        t0 = time.time()
        z = np.load(OBS)
        self.npz = z
        self.image = torch.from_numpy(z["image"]).to(dev)
        self.wrist = torch.from_numpy(z["wrist_image"]).to(dev)
        self.state = torch.from_numpy(np.load(DATA / "state_norm.npy")).to(dev)
        self.tok = torch.from_numpy(np.load(DATA / "tokenized_prompt.npy")).to(dev)
        self.tokm = torch.from_numpy(np.load(DATA / "tokenized_prompt_mask.npy")).to(dev)
        self.n = len(self.image)
        self.cfg = cfg
        self.dev = dev
        print(f"frames {self.n} loaded in {time.time() - t0:.0f}s", flush=True)

    def base(self, idx):
        b = len(idx)
        ones = torch.ones(b, dtype=torch.bool, device=self.dev)
        return _model.Observation.from_dict({
            "image": {"base_0_rgb": self.image[idx], "left_wrist_0_rgb": self.wrist[idx],
                      "right_wrist_0_rgb": torch.zeros_like(self.image[idx])},
            "image_mask": {"base_0_rgb": ones, "left_wrist_0_rgb": ones, "right_wrist_0_rgb": ~ones},
            "state": self.state[idx], "tokenized_prompt": self.tok[None].expand(b, -1),
            "tokenized_prompt_mask": self.tokm[None].expand(b, -1)})

    def proxy(self, idx):
        """What their proxy policy's input pipeline gives: the base's two cameras ([-1, 1] float via
        Observation.from_dict), the normalized state's real dims."""
        b = len(idx)
        ones = torch.ones(b, dtype=torch.bool, device=self.dev)
        return _model.Observation.from_dict({
            "image": {"base_0_rgb": self.image[idx], "left_wrist_0_rgb": self.wrist[idx]},
            "image_mask": {"base_0_rgb": ones, "left_wrist_0_rgb": ones},
            "state": self.state[idx, : self.cfg.state_dim]})


@torch.no_grad()
def build_prefix_cache(model, frames, bs=16):
    """Their prefix code (forward_for_distill) per frame; keep valid positions only. Returns pinned host K, V
    (F, layers, Lv, head_dim) bf16."""
    K = V = None
    valid0 = None
    t0 = time.time()
    for i in range(0, frames.n, bs):
        idx = torch.arange(i, min(frames.n, i + bs), device=frames.dev)
        obs = frames.base(idx)
        images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(obs, train=False)  # noqa: SLF001
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)  # noqa: SLF001
        model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        _, pkv = model.paligemma_with_expert.forward(attention_mask=prefix_att_2d_masks_4d,
                                                     position_ids=prefix_position_ids, past_key_values=None,
                                                     inputs_embeds=[prefix_embs, None], use_cache=True)
        valid = prefix_pad_masks[0]
        assert bool((prefix_pad_masks == valid[None]).all())
        if K is None:
            valid0 = valid.clone()
            legacy = pkv.to_legacy_cache()
            nl, (_, nkv, _, hd) = len(legacy), legacy[0][0].shape
            assert nkv == 1
            lv = int(valid.sum())
            K = torch.empty((frames.n, nl, lv, hd), dtype=torch.bfloat16, pin_memory=True)
            V = torch.empty((frames.n, nl, lv, hd), dtype=torch.bfloat16, pin_memory=True)
            print(f"prefix cache: {nl} layers, {lv} valid of {valid.numel()} prefix tokens, "
                  f"{2 * K.numel() * 2 / 1e9:.1f} GB pinned host", flush=True)
        assert torch.equal(valid, valid0)
        legacy = pkv.to_legacy_cache()
        ks = torch.stack([k[:, 0][:, valid] for k, _ in legacy], 1)
        vs = torch.stack([v[:, 0][:, valid] for _, v in legacy], 1)
        K[i:i + len(idx)] = ks.cpu()
        V[i:i + len(idx)] = vs.cpu()
        if i % 1600 == 0:
            print(f"  cache {i}/{frames.n} {time.time() - t0:.0f}s", flush=True)
    return K, V, valid0


def cache_for(K, V, idx_cpu, dev):
    k = K[idx_cpu].to(dev, non_blocking=True)
    v = V[idx_cpu].to(dev, non_blocking=True)
    return k, v


def make_cache(k, v):
    return DynamicCache.from_legacy_cache(tuple((k[:, l][:, None], v[:, l][:, None]) for l in range(k.shape[1])))


@torch.no_grad()
def teacher_labels(model, frames, idx, kv, num_steps=10):
    """Their forward_for_distill with the prefix cache precomputed (see module doc)."""
    obs = frames.base(idx)
    state = obs.state  # pi05_libero's padded normalized state; unused by the pi0.5 suffix
    state = torch.nn.functional.pad(state, (0, model.config.action_dim - state.shape[1]), mode="constant", value=0)
    bsize, device = state.shape[0], state.device
    actions_shape = (bsize, model.config.action_horizon, model.config.action_dim)
    initial_noise = model.sample_noise(actions_shape, device)
    time_schedule = sample_bin_times(bsize, num_steps, device)
    pkv = make_cache(*kv)
    prefix_pad_masks = torch.ones(bsize, kv[0].shape[2], dtype=torch.bool, device=device)
    return _their_flow_path(model, state, prefix_pad_masks, pkv, initial_noise, time_schedule, bsize, device)


def student_losses(net, observation, noises, times, gradients):
    """Their ProxyPytorch.forward_distill (proxy_pytorch.py, use_noise=True) with the image tokens run once per
    observation (see module doc). Returns per element losses (B, num_steps - 1, H, A)."""
    cfg = net.config
    images, img_masks, state = net._preprocess_observation(observation, train=True)  # noqa: SLF001
    prefix_embs, prefix_pad_masks, _ = net.embed_prefix(images, img_masks)
    noises = noises[:, 1:, :, : cfg.action_dim]
    times = times[:, 1:]
    gradients = gradients[:, 1:, :, : cfg.action_dim]
    batch_size, num_steps = times.shape[:2]
    flat_times = times.reshape(batch_size * num_steps)
    flat_gradients = gradients.reshape(batch_size * num_steps, cfg.action_horizon, cfg.action_dim)
    flat_x_t = noises.reshape(batch_size * num_steps, cfg.action_horizon, cfg.action_dim)
    flat_state = state[:, None, :].expand(batch_size, num_steps, state.shape[-1])
    flat_state = flat_state.reshape(batch_size * num_steps, state.shape[-1])
    suffix_embs, suffix_pad_masks, _, adarms_cond = net.embed_suffix(flat_state, flat_x_t, flat_times)
    assert bool(prefix_pad_masks.all()) and bool(suffix_pad_masks.all())
    gm = net.expert_model.gemma_expert.model
    dev = prefix_embs.device
    P, Ls = prefix_embs.shape[1], suffix_embs.shape[1]
    out_p = gm.forward(inputs_embeds=prefix_embs, attention_mask=prefix_pad_masks,
                       position_ids=torch.arange(P, device=dev)[None].expand(batch_size, P),
                       past_key_values=None, use_cache=True, adarms_cond=None)
    legacy = tuple((k.repeat_interleave(num_steps, 0), v.repeat_interleave(num_steps, 0))
                   for k, v in out_p.past_key_values.to_legacy_cache())
    out_s = gm.forward(inputs_embeds=suffix_embs,
                       attention_mask=torch.ones(batch_size * num_steps, P + Ls, dtype=torch.bool, device=dev),
                       position_ids=(P + torch.arange(Ls, device=dev))[None].expand(batch_size * num_steps, Ls),
                       past_key_values=DynamicCache.from_legacy_cache(legacy), use_cache=False,
                       adarms_cond=adarms_cond)
    suffix_out = out_s.last_hidden_state[:, -cfg.action_horizon:].to(dtype=torch.float32)
    v_t = net.action_out_proj(suffix_out)
    loss = F.mse_loss(v_t, flat_gradients, reduction="none")
    return loss.reshape(batch_size, num_steps, cfg.action_horizon, cfg.action_dim)


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def _uncached_prefix(model, obs):
    """Their forward_for_distill's prefix code, verbatim (returns state, pad masks, cache)."""
    images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(obs, train=False)  # noqa: SLF001
    state = torch.nn.functional.pad(state, (0, model.config.action_dim - state.shape[1]), mode="constant", value=0)
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)  # noqa: SLF001
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
    _, pkv = model.paligemma_with_expert.forward(attention_mask=prefix_att_2d_masks_4d,
                                                 position_ids=prefix_position_ids, past_key_values=None,
                                                 inputs_embeds=[prefix_embs, None], use_cache=True)
    return state, prefix_pad_masks, pkv


@torch.no_grad()
def check_teacher(model, frames, K, V, dev):
    """Cached + compacted prefix vs their uncached prefix, with explicit identical noise and times. The noise floor is
    the same uncached teacher run as 2 batches of 4 instead of 1 batch of 8 (bf16 kernels differ with batch shape)."""
    idx = torch.arange(0, frames.n, frames.n // 8, device=dev)[:8]
    g = torch.Generator(device=dev).manual_seed(1234)
    noise = torch.randn((8, model.config.action_horizon, model.config.action_dim), generator=g, device=dev)
    sched = (torch.arange(10, device=dev) / 10 + torch.rand((8, 10), generator=g, device=dev) / 10).flip(1)
    st, pm, pkv = _uncached_prefix(model, frames.base(idx))
    full8 = _their_flow_path(model, st, pm, pkv, noise, sched, 8, dev)
    parts = []
    for h in (slice(0, 4), slice(4, 8)):
        st, pm, pkv = _uncached_prefix(model, frames.base(idx[h]))
        parts.append(_their_flow_path(model, st, pm, pkv, noise[h], sched[h], 4, dev))
    full4 = [torch.cat([parts[0][i], parts[1][i]]) for i in range(4)]
    k, v = cache_for(K, V, idx.cpu(), dev)
    st = frames.base(idx).state
    cached = _their_flow_path(model, st, torch.ones(8, k.shape[2], dtype=torch.bool, device=dev), make_cache(k, v),
                              noise, sched, 8, dev)

    def cmp(a, b):
        cos = F.cosine_similarity(a[2][..., :7].flatten(2).double(), b[2][..., :7].flatten(2).double(), dim=2)
        return {"v_base_rel_l2": rel(a[2], b[2]), "v_base_real_rel_l2": rel(a[2][..., :7], b[2][..., :7]),
                "v_base_real_cos_min": float(cos.min()), "x_t_path_rel_l2": rel(a[0], b[0]),
                "final_actions_real_rel_l2": rel(a[3][..., :7], b[3][..., :7])}
    out = {"frames": idx.tolist(), "cached_vs_full_batch8": cmp(cached, full8),
           "noise_floor_full_batch4_vs_batch8": cmp(full4, full8)}
    c, f = out["cached_vs_full_batch8"], out["noise_floor_full_batch4_vs_batch8"]
    # pass if the cache moves the teacher no more than a batch-shape change of the uncached teacher does (x2 + slack)
    out["pass"] = bool(c["v_base_real_rel_l2"] <= 2 * f["v_base_real_rel_l2"] + 5e-3
                       and c["final_actions_real_rel_l2"] <= 2 * f["final_actions_real_rel_l2"] + 2e-3
                       and c["x_t_path_rel_l2"] <= 2 * f["x_t_path_rel_l2"] + 1e-3)
    return out


def check_student(net, frames, model, K, V, dev):
    """Prefix-once student vs their forward_distill, same RNG state (same augmentation draw), fp32 "highest".
    Noise floor: their forward_distill run twice with the same RNG state (fp32 attention backward is not bitwise
    deterministic)."""
    idx = torch.arange(3, frames.n, frames.n // 8, device=dev)[:8]
    with torch.no_grad():
        noises, times, grads, acts = teacher_labels(model, frames, idx, cache_for(K, V, idx.cpu(), dev))
    obs = frames.proxy(idx)

    def run(fn):
        torch.manual_seed(77)
        l = fn()
        l.mean().backward()
        gr = {n: q.grad.detach().clone() for n, q in net.named_parameters() if q.grad is not None}
        net.zero_grad(set_to_none=True)
        return l.detach(), gr

    with official.precision("highest"):
        net.train()
        l1, g1 = run(lambda: net(obs, acts, noises=noises, times=times, gradients=grads, mode="distill"))
        l1b, g1b = run(lambda: net(obs, acts, noises=noises, times=times, gradients=grads, mode="distill"))
        l2, g2 = run(lambda: student_losses(net, obs, noises, times, grads))
    assert set(g1) == set(g2) == set(g1b), set(g1) ^ set(g2)

    def stats(ga, gb, la, lb):
        cos = {n: float(F.cosine_similarity(ga[n].flatten().double(), gb[n].flatten().double(), dim=0))
               for n in gb if gb[n].abs().max() > 0}
        worst = min(cos, key=cos.get)
        return {"loss_rel_l2": rel(la, lb),
                "grad_rel_l2_all": rel(torch.cat([ga[n].flatten() for n in gb]), torch.cat([gb[n].flatten() for n in gb])),
                "grad_cos_min": cos[worst], "grad_cos_min_param": worst}
    out = {"loss_shape": list(l1.shape), "loss_their": float(l1.mean()), "loss_ours": float(l2.mean()),
           "n_params_with_grad": len(g1), "precision": "highest",
           "ours_vs_their": stats(g2, g1, l2, l1), "noise_floor_their_vs_their": stats(g1b, g1, l1b, l1)}
    o, f = out["ours_vs_their"], out["noise_floor_their_vs_their"]
    out["pass"] = bool(o["loss_rel_l2"] < 1e-5 and o["grad_rel_l2_all"] <= max(3 * f["grad_rel_l2_all"], 2e-3)
                       and o["grad_cos_min"] >= f["grad_cos_min"] - 0.01)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0, help="seed index; torch/numpy seed = 42 + seed (theirs is 42)")
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--save-every", type=int, default=10_000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--probe-every", type=int, default=1000)
    p.add_argument("--checks-only", action="store_true")
    args = p.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=False if not args.checks_only else True)
    dev = torch.device("cuda")
    seed = 42 + args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    info = json.loads((DATA / "real_dims.json").read_text())
    cfg = official.ProxyCfg(action_dim=info["action_dim_real"], state_dim=info["state_dim_real"],
                            action_horizon=info["action_horizon"])

    t0 = time.time()
    sp = _policy.SteerablePolicy(_policy.load_openpi_policy(compile_mode=None))
    model = sp.model
    model.eval()
    for q in model.parameters():
        q.requires_grad = False
    print(f"teacher loaded in {time.time() - t0:.0f}s on {torch.cuda.get_device_name()}", flush=True)
    frames = Frames(dev, cfg)
    # Observation built here == the policy's own input transform (checked on a few frames)
    z = frames.npz
    for f in (0, 1234, frames.n - 1):
        o = {"observation/image": z["image"][f], "observation/wrist_image": z["wrist_image"][f],
             "observation/state": z["state"][f].astype(np.float64),
             "prompt": str(z["prompt"])}
        _, ob = sp.observation(o)
        mine = frames.base(torch.tensor([f], device=dev))
        for k in ob.images:
            assert torch.equal(ob.images[k], mine.images[k]) and torch.equal(ob.image_masks[k], mine.image_masks[k]), k
        # the client sends a float64 state; their proxy casts it to float32 (embed_suffix), the teacher ignores it
        assert torch.equal(ob.state.float(), mine.state), (ob.state, mine.state)
        assert torch.equal(ob.tokenized_prompt, mine.tokenized_prompt)
        assert torch.equal(ob.tokenized_prompt_mask, mine.tokenized_prompt_mask)
    del z, frames.npz
    torch.set_float32_matmul_precision("high")  # their ProxyPytorch.__init__ sets this for the whole training process
    K, V, _ = build_prefix_cache(model, frames)
    print(f"prefix cache built at {time.time() - t0:.0f}s", flush=True)

    net = official.build_proxy(cfg).to(dev)
    torch.set_float32_matmul_precision("high")
    n_params = sum(q.numel() for q in net.parameters())
    checks = {"teacher_cache": check_teacher(model, frames, K, V, dev)}
    print("check (a) teacher:", json.dumps(checks["teacher_cache"]), flush=True)
    checks["student_prefix_once"] = check_student(net, frames, model, K, V, dev)
    print("check (b) student:", json.dumps(checks["student_prefix_once"]), flush=True)
    ok = checks["teacher_cache"]["pass"] and checks["student_prefix_once"]["pass"]
    checks["pass"] = bool(ok)
    (out / "checks.json").write_text(json.dumps(checks, indent=1))
    if not ok:
        print("EQUIVALENCE CHECK FAILED", flush=True)
        sys.exit(3)
    if args.checks_only:
        return
    # fresh init again so the checks' backward passes leave no trace (same seed as a run without checks)
    torch.manual_seed(seed)
    net = official.build_proxy(cfg).to(dev)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(seed + 1)
    np.random.seed(seed + 1)

    # fixed probe (monitoring only): 64 frames, fixed noise and times, student in eval mode (no augmentation)
    probe_idx = torch.arange(0, frames.n, frames.n // 64, device=dev)[:64]
    g = torch.random.fork_rng(devices=[dev])
    with g:
        torch.manual_seed(999)
        probe_lab = teacher_labels(model, frames, probe_idx, cache_for(K, V, probe_idx.cpu(), dev))
    probe_obs = frames.proxy(probe_idx)

    @torch.no_grad()
    def probe():
        net.eval()
        noises, times, grads, _ = probe_lab
        with torch.random.fork_rng(devices=[dev]):
            l = student_losses_eval(net, probe_obs, noises, times, grads)
        net.train()
        return l

    def student_losses_eval(net_, obs, noises, times, grads):
        # eval mode: their preprocess(train=True) would still augment; use their sample-time path instead:
        # same expert call as their eval_steering._sequence_proxy_flow_from_prefix on each path point
        images, img_masks, state = net_._preprocess_observation(obs, train=False)  # noqa: SLF001
        prefix_embs, prefix_pad_masks, _ = net_.embed_prefix(images, img_masks)
        A = cfg.action_dim
        vs = []
        for k in range(1, noises.shape[1]):
            suffix_embs, suffix_pad_masks, _, adarms = net_.embed_suffix(state, noises[:, k, :, :A], times[:, k])
            embs = torch.cat([prefix_embs, suffix_embs], 1)
            pad = torch.cat([prefix_pad_masks, suffix_pad_masks], 1)
            pos = (torch.cumsum(pad, dim=1) - 1).to(torch.long)
            h, _ = net_.expert_model.forward(attention_mask=pad, position_ids=pos, past_key_values=None,
                                             inputs_embeds=embs, use_cache=False, adarms_cond=adarms)
            vs.append(net_.action_out_proj(h[:, -cfg.action_horizon:].to(torch.float32)))
        v = torch.stack(vs, 1)
        tgt = grads[:, 1:, :, :A]
        cos = F.cosine_similarity(v.flatten(2).double(), tgt.flatten(2).double(), dim=2)
        return {"mse_real": float(((v - tgt) ** 2).mean()), "cos_real_mean": float(cos.mean()),
                "cos_real_by_point": cos.mean(0).tolist()}

    optim = torch.optim.AdamW(net.parameters(), lr=2.5e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10)
    commit = subprocess.run(["git", "-C", str(pathlib.Path(__file__).resolve().parents[2]), "rev-parse", "--short",
                             "HEAD"], capture_output=True, text=True).stdout.strip()
    config = dict(vars(args), torch_seed=seed, proxy_cfg=cfg.__dict__, n_params=n_params,
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__, repo_commit=commit,
                  official_commit=official.COMMIT, official_sources=official.SOURCES,
                  teacher="/workspace/checkpoints/pi05_libero_pytorch (pi05_libero, eager, bf16 as loaded)",
                  num_distill_steps=10, use_noise_for_distill=True, teacher_flow_path_noise_std=0.0,
                  optimizer=dict(name="AdamW", betas=[0.9, 0.95], eps=1e-8, weight_decay=1e-10, clip=1.0),
                  lr_schedule=dict(warmup=1000, peak=2.5e-5, decay_steps=30000, end=2.5e-6),
                  float32_matmul_precision="high", released_metadata="pot/weight reference/20000: batch_size 8, "
                  "num_train_steps 20001, seed 42, num_distill_steps 10, use_noise_for_distill True",
                  data=dict(frames=frames.n, obs=OBS, state=str(DATA / "state_norm.npy")), checks=checks)
    (out / "config.json").write_text(json.dumps(config, indent=1, default=str))

    hist = {"train": [], "probe": [dict(step=0, **probe())]}
    print("probe step 0:", hist["probe"][-1]["mse_real"], hist["probe"][-1]["cos_real_mean"], flush=True)
    rng = np.random.default_rng(seed)
    perm, pos = rng.permutation(frames.n), 0
    pool = concurrent.futures.ThreadPoolExecutor(1)

    def next_group():
        nonlocal perm, pos
        ids = []
        for _ in range(GROUP):
            if pos + args.batch > frames.n:
                perm, pos = rng.permutation(frames.n), 0
            ids.append(perm[pos:pos + args.batch])
            pos += args.batch
        flat = torch.from_numpy(np.concatenate(ids))
        return flat, (K[flat], V[flat])

    fut = pool.submit(next_group)
    net.train()
    t_start = time.time()
    run = []
    step = 0
    while step < args.steps:
        flat, (kc, vc) = fut.result()
        fut = pool.submit(next_group)
        idx = flat.to(dev)
        kv = (kc.to(dev, non_blocking=True), vc.to(dev, non_blocking=True))
        noises, times, grads, acts = teacher_labels(model, frames, idx, kv)
        for j in range(GROUP):
            if step >= args.steps:
                break
            sl = slice(j * args.batch, (j + 1) * args.batch)
            for pg in optim.param_groups:
                pg["lr"] = their_lr_schedule(step)
            losses = student_losses(net, frames.proxy(idx[sl]), noises[sl], times[sl], grads[sl])
            loss = losses.mean()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optim.step()
            optim.zero_grad(set_to_none=True)
            run.append((loss.detach(), gn.detach()))
            step += 1
            if step % args.log_every == 0:
                el = time.time() - t_start
                ls = float(torch.stack([a for a, _ in run]).mean())
                gs = float(torch.stack([b for _, b in run]).mean())
                run = []
                hist["train"].append(dict(step=step, loss=ls, grad_norm=gs, lr=their_lr_schedule(step - 1),
                                          elapsed_s=el))
                print(f"step {step} loss {ls:.5f} grad_norm {gs:.3f} lr {their_lr_schedule(step - 1):.2e} "
                      f"{el / step * 1000:.0f} ms/step eta {el / step * (args.steps - step) / 60:.1f} min "
                      f"mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
            if step % args.probe_every == 0 or step == args.steps:
                hist["probe"].append(dict(step=step, elapsed_s=time.time() - t_start, **probe()))
                print(f"probe step {step}: mse {hist['probe'][-1]['mse_real']:.5f} "
                      f"cos {hist['probe'][-1]['cos_real_mean']:.4f}", flush=True)
                (out / "metrics.json").write_text(json.dumps(hist, indent=1))
            if step % args.save_every == 0 or step == args.steps:
                name = "final.pt" if step == args.steps else f"step_{step}.pt"
                official.save_proxy(out / name, net, cfg, step=step, kind="reference", seed=args.seed,
                                    gpu=torch.cuda.get_device_name())
                print(f"saved {name}", flush=True)
    hist["train_s"] = time.time() - t_start
    hist["gpu"] = torch.cuda.get_device_name()
    (out / "metrics.json").write_text(json.dumps(hist, indent=1))
    print(f"done in {hist['train_s'] / 60:.1f} min on {hist['gpu']}", flush=True)


if __name__ == "__main__":
    main()
