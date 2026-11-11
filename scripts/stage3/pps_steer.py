"""PPS steering hook for pps_real.sampler with the paper style proxies. Runs in the openpi venv.

v = v_base + gamma * (v_task - v_ref) at each of the 10 Euler steps (PPS Eq. 4).

- The proxies read the two 224 x 224 uint8 frames the base sees (observation/image = agentview, observation/
  wrist_image = wrist), which are exactly the arrays pps_real.libero_obs.process_images makes and the ones the proxies
  were trained on (/workspace/data/t59_demos/policy_obs.npz was rendered with the same function). Camera order and
  normalization are PaperProxy.encode's, as in training.
- Images are encoded once per inference call (begin), and the image key/value cache is reused for all 10 steps.
- Proxies run under bf16 autocast as in training and in check_task_paper.py, on the sampler's float32 x_t and t.
- Unlike pps_real.sampler.pps_hook, gamma = 0 does not short circuit. Both proxies are still evaluated and the hook
  returns v_base + 0 * (v_task - v_ref), so the gamma = 0 check exercises the whole PPS code path.
- Per step velocity cosines (float64, over the flattened chunk, as check_task_paper.py) are kept for the call:
  ref vs base, task vs ref, task vs base, each over the first 7 dims and all 32, plus |v_base| and |gamma*delta|.
"""

import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "stage2"))
from paper_proxy import load_proxy  # noqa: E402

STAT_KEYS = ("ref_base_first7", "ref_base_all32", "task_ref_first7", "task_ref_all32", "task_base_first7",
             "task_base_all32", "norm_vbase_all32", "norm_delta_all32", "norm_vbase_first7", "norm_delta_first7")


def _cos(a, b):
    return F.cosine_similarity(a.flatten(1).double(), b.flatten(1).double(), dim=1)


class PPSSteer:
    def __init__(self, ref_path, task_path, gamma, device="cuda"):
        self.device = torch.device(device)
        self.ref = load_proxy(ref_path, self.device)
        self.task = load_proxy(task_path, self.device)
        self.gamma = float(gamma)
        self.kv_r = self.kv_t = None
        self._stats = None

    @torch.no_grad()
    def begin(self, image: np.ndarray, wrist: np.ndarray):
        assert image.dtype == np.uint8 and image.shape == (224, 224, 3), (image.dtype, image.shape)
        assert wrist.dtype == np.uint8 and wrist.shape == (224, 224, 3), (wrist.dtype, wrist.shape)
        imgs = torch.from_numpy(np.stack([image, wrist])[None]).to(self.device)  # (1, 2, 224, 224, 3)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self.kv_r = self.ref.image_kv(self.ref.encode(imgs))
            self.kv_t = self.task.image_kv(self.task.encode(imgs))
        self._stats = []

    @torch.no_grad()
    def __call__(self, step, x_t, t, v_base, ctx):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vr = self.ref.velocity(self.kv_r, x_t, t).float()
            vt = self.task.velocity(self.kv_t, x_t, t).float()
        delta = self.gamma * (vt - vr)
        v = v_base + delta
        vb = v_base.float()
        self._stats.append(torch.stack([
            _cos(vr[..., :7], vb[..., :7]), _cos(vr, vb), _cos(vt[..., :7], vr[..., :7]), _cos(vt, vr),
            _cos(vt[..., :7], vb[..., :7]), _cos(vt, vb),
            vb.flatten(1).double().norm(dim=1), delta.flatten(1).double().norm(dim=1),
            vb[..., :7].flatten(1).double().norm(dim=1), delta[..., :7].flatten(1).double().norm(dim=1),
        ], 1)[0])
        return v

    def stats(self):
        """(10, len(STAT_KEYS)) float64 for the last call, as a dict of 10 vectors."""
        s = torch.stack(self._stats).cpu().numpy()
        return {k: s[:, i] for i, k in enumerate(STAT_KEYS)}


# ---------------------------------------------------------------- task proxy alone (added for the 4090 rerun, step 6)
# Definition used: the task proxy as a standalone policy. Its velocity replaces the base's on all 32 dims at all 10
# Euler steps, so the actions come from the task proxy only (no base velocity, no reference proxy, no gamma). This is
# the analogue of the authors' --only_steer for a proxy that predicts all 32 dims: their proxy reads and predicts only
# the real action dims, so their actions also come from the task proxy alone. Same noise, time grid (openpi's float32
# t = 1.0, 0.9, ... accumulated as in pps_real.sampler) and output transform as the base.

class TaskOnlyHook:
    """Reference implementation: run through pps_real.sampler with the base evaluated but ignored. Uses the task proxy
    of a PPSSteer (same weights, same image encoding code as PPSSteer.begin)."""

    def __init__(self, steer):
        self.s = steer
        self.kv_t = None

    @torch.no_grad()
    def begin(self, image, wrist):
        self.kv_t = encode_task(self.s, image, wrist)

    @torch.no_grad()
    def __call__(self, step, x_t, t, v_base, ctx):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.s.task.velocity(self.kv_t, x_t, t).float()


@torch.no_grad()
def encode_task(steer, image, wrist):
    assert image.dtype == np.uint8 and image.shape == (224, 224, 3), (image.dtype, image.shape)
    assert wrist.dtype == np.uint8 and wrist.shape == (224, 224, 3), (wrist.dtype, wrist.shape)
    imgs = torch.from_numpy(np.stack([image, wrist])[None]).to(steer.device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return steer.task.image_kv(steer.task.encode(imgs))


@torch.no_grad()
def task_only_infer(sp, steer, obs, noise):
    """Fast path: the task proxy's own Euler loop, the base is not run. Mirrors SteerablePolicy.infer (input transform
    for the state the output transform carries, batch 1) and pps_real.sampler's loop arithmetic exactly, so it must be
    bit identical to sp.infer(obs, noise=noise, hook=TaskOnlyHook(steer)); check_task_only.py checks that."""
    import jax
    inputs, _ = sp.observation(obs)
    kv = encode_task(steer, np.asarray(obs["observation/image"]), np.asarray(obs["observation/wrist_image"]))
    x_t = torch.from_numpy(noise).to(sp.device)
    if x_t.ndim == 2:
        x_t = x_t[None, ...]
    dt = torch.tensor(-1.0 / sp.num_steps, dtype=torch.float32, device=sp.device)
    time = torch.tensor(1.0, dtype=torch.float32, device=sp.device)
    while time >= -dt / 2:
        expanded_time = time.expand(x_t.shape[0])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v_t = steer.task.velocity(kv, x_t, expanded_time.clone()).float()
        x_t = x_t + dt * v_t
        time += dt
    outputs = {"state": inputs["state"], "actions": x_t}
    outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
    outputs = sp.policy._output_transform(outputs)  # noqa: SLF001
    return outputs
