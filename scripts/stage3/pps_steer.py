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
