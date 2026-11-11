"""Steering with the faithful proxies inside pps_real.sampler (pi0.5's own eager sampler, unchanged). openpi venv.

Per Euler step, with v_base = pi0.5's velocity at the shared x_t and t, A = 7 real action dims (norm stats):
  pps     v = v_base.clone(); v[..., :A] += gamma * (v_task - v_ref)              their eval_steering.py:485-489
  blend   v = v_base.clone(); v[..., :A] = (1 - gamma) * v_base[..., :A] + gamma * v_task
  only    v = v_base.clone(); v[..., :A] = v_task  (their --only_steer, eval_steering.py:481-484): the base still runs
          and its velocity drives the 25 padding dims; the real dims come from the task proxy alone.
Padding dims (A..31) always keep v_base unchanged; every step asserts that bit for bit and counts it.

Proxies see what their eval gives them (_prepare_proxy_steering + _sequence_proxy_flow_from_prefix, copied below):
their _preprocess_observation(train=False) and embed_prefix once per call on the base's two cameras ([-1, 1] float,
Observation.from_dict of the client's uint8 frames), the base's normalized state's real dims for the state token, and
x_t[..., :A] with the base's t at every step. Proxy calls run under float32_matmul_precision "high" (TF32, as their
ProxyPytorch.__init__ sets for their eval) inside a save/restore context; pi0.5 runs under the process setting, which
upstream openpi's PI0Pytorch.__init__ itself sets to "high" (pi0_pytorch.py:111), so its numerics are exactly the
stock eager sampler's. gamma = 0 is not short circuited.

task_only_fast(): the "only" rule without running the base. The real dims of x_t evolve by v_task alone and the
proxy reads only x_t[..., :A], so the real-dim actions do not depend on the base at all; the fast path reproduces the
hook path's real-dim output bit for bit (checked in check_faithful.py before any eval). The 25 padding dims of the
fast path's raw output are left at the noise (they never reach the robot: LiberoOutputs keeps [:7]).
"""

import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import official  # noqa: E402
from openpi.models import model as _model  # noqa: E402

STAT_KEYS = ("ref_base_real", "task_ref_real", "task_base_real", "norm_vbase_real", "norm_delta_real")


# ---- copied from their eval_steering.py (a22f9ad) ----
def _prepare_proxy_steering(model, observation):
    images, img_masks, state = model._preprocess_observation(observation, train=False)
    prefix_embs, prefix_pad_masks, _ = model.embed_prefix(images, img_masks)
    return {
        "kind": "sequence",
        "state": state,
        "prefix_embs": prefix_embs,
        "prefix_pad_masks": prefix_pad_masks,
    }


def _sequence_proxy_flow_from_prefix(
    model,
    state,
    prefix_embs,
    prefix_pad_masks,
    x_t_path,
    time_cond,
    action_dim: int,
):
    x_t_model = x_t_path[:, :, :action_dim]
    suffix_embs, suffix_pad_masks, _, adarms_cond = model.embed_suffix(
        state,
        x_t_model,
        time_cond,
    )
    embs = torch.cat([prefix_embs, suffix_embs], dim=1)
    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    position_ids = (torch.cumsum(pad_masks, dim=1) - 1).to(dtype=torch.long)

    hidden_states, _ = model.expert_model.forward(
        attention_mask=pad_masks,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=embs,
        use_cache=False,
        adarms_cond=adarms_cond,
    )
    suffix_out = hidden_states[:, -model.config.action_horizon :]
    suffix_out = suffix_out.to(dtype=torch.float32)
    return model.action_out_proj(suffix_out)
# ---- end copy ----


def _cos(a, b):
    return F.cosine_similarity(a.flatten(1).double(), b.flatten(1).double(), dim=1)


def proxy_observation(image, wrist, state_norm, state_dim, device):
    """The proxy policy's input: base cameras as openpi Observation.from_dict makes them, state's real dims."""
    im = torch.from_numpy(np.ascontiguousarray(image))[None].to(device)
    wr = torch.from_numpy(np.ascontiguousarray(wrist))[None].to(device)
    ones = torch.ones(1, dtype=torch.bool, device=device)
    st = torch.as_tensor(state_norm, device=device)[..., :state_dim]
    if st.ndim == 1:
        st = st[None]
    return _model.Observation.from_dict({"image": {"base_0_rgb": im, "left_wrist_0_rgb": wr},
                                         "image_mask": {"base_0_rgb": ones, "left_wrist_0_rgb": ones}, "state": st})


class FaithfulSteer:
    """Velocity hook for pps_real.sampler. Call begin() once per policy call, then pass self as the hook."""

    def __init__(self, ref_path, task_path, device="cuda"):
        self.device = torch.device(device)
        self.ref, self.cfg, self.ref_meta = official.load_proxy(ref_path, self.device)
        self.task, cfg_t, self.task_meta = official.load_proxy(task_path, self.device)
        assert cfg_t == self.cfg
        self.A = self.cfg.action_dim
        self.mode, self.gamma = "pps", 0.0
        self.prep_r = self.prep_t = None
        self._stats = []
        self.pad_checked = 0
        self.pad_violations = 0

    @torch.no_grad()
    def begin(self, image, wrist, state_norm, mode, gamma):
        assert mode in ("pps", "blend", "only"), mode
        assert image.dtype == np.uint8 and image.shape == (224, 224, 3), (image.dtype, image.shape)
        assert wrist.dtype == np.uint8 and wrist.shape == (224, 224, 3), (wrist.dtype, wrist.shape)
        self.mode, self.gamma = mode, float(gamma)
        obs = proxy_observation(image, wrist, state_norm, self.cfg.state_dim, self.device)
        with official.precision("high"):
            self.prep_t = _prepare_proxy_steering(self.task, obs)
            self.prep_r = _prepare_proxy_steering(self.ref, obs) if mode == "pps" else None
        self._stats = []

    def _flow(self, model, prep, x_t, t):
        return _sequence_proxy_flow_from_prefix(model, prep["state"], prep["prefix_embs"], prep["prefix_pad_masks"],
                                                x_t, t, self.A)

    @torch.no_grad()
    def __call__(self, step, x_t, t, v_base, ctx):
        A = self.A
        with official.precision("high"):
            vt = self._flow(self.task, self.prep_t, x_t, t)
            vr = self._flow(self.ref, self.prep_r, x_t, t) if self.mode == "pps" else None
        v = v_base.clone()
        if self.mode == "pps":
            v[:, :, :A] += self.gamma * (vt - vr)
        elif self.mode == "blend":
            v[:, :, :A] = (1.0 - self.gamma) * v_base[:, :, :A] + self.gamma * vt
        else:
            v[:, :, :A] = vt
        self.pad_checked += 1
        if not torch.equal(v[:, :, A:], v_base[:, :, A:]):
            self.pad_violations += 1
        vb = v_base[:, :, :A].float()
        d = (v[:, :, :A] - v_base[:, :, :A]).float()
        nan = torch.full_like(_cos(vt, vb), float("nan"))
        self._stats.append(torch.stack([
            _cos(vr, vb) if vr is not None else nan, _cos(vt, vr) if vr is not None else nan, _cos(vt, vb),
            vb.flatten(1).double().norm(dim=1), d.flatten(1).double().norm(dim=1)], 1)[0])
        return v

    def stats(self):
        s = torch.stack(self._stats).cpu().numpy()
        return {k: s[:, i] for i, k in enumerate(STAT_KEYS)}


@torch.no_grad()
def task_only_fast(sp, steer, obs, noise):
    """The 'only' rule without the base (see module doc). Mirrors SteerablePolicy.infer's transforms and
    pps_real.sampler's loop arithmetic (float32 time accumulated in place, dt = -1/num_steps)."""
    import jax
    inputs, _ = sp.observation(obs)
    A = steer.A
    pobs = proxy_observation(np.asarray(obs["observation/image"]), np.asarray(obs["observation/wrist_image"]),
                             inputs["state"][0], steer.cfg.state_dim, sp.device)
    x_full = torch.from_numpy(noise).to(sp.device)
    if x_full.ndim == 2:
        x_full = x_full[None, ...]
    with official.precision("high"):
        prep = _prepare_proxy_steering(steer.task, pobs)
        x_t = x_full
        dt = torch.tensor(-1.0 / sp.num_steps, dtype=torch.float32, device=sp.device)
        time = torch.tensor(1.0, dtype=torch.float32, device=sp.device)
        while time >= -dt / 2:
            expanded_time = time.expand(x_t.shape[0])
            vt = steer._flow(steer.task, prep, x_t, expanded_time.clone())
            v = torch.zeros_like(x_t)
            v[:, :, :A] = vt
            x_t = x_t + dt * v
            time += dt
    outputs = {"state": inputs["state"], "actions": x_t}
    outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
    return sp.policy._output_transform(outputs)  # noqa: SLF001
