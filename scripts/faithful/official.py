"""Load the official PPS proxy code (github.com/TritiumR/pps @ a22f9ad) from /workspace/external/pps_official, unchanged.

Their proxy lives in their fork's package `openpi`, which clashes with our openpi (upstream 215abfb, the pi0.5 code path
that must not change). So each of their modules below is executed from its own source file under a private module name
(pps_official.*). The only edit is the import lines that point into their package, which are redirected to the
corresponding pps_official.* module. Every other line is theirs, byte for byte; the file hashes are recorded in
SOURCES so a run can log exactly what it executed.

Modules
  pps_official.gemma                 their models/gemma.py (for get_config("gemma_12m")); its own imports of
                                     openpi.models.lora / openpi.shared / openpi.training resolve to our openpi.
  pps_official.preprocessing_pytorch their models_pytorch/preprocessing_pytorch.py (augmentation; imports our
                                     openpi.shared.image_tools, which only matters for non 224 images).
  pps_official.expert_pytorch        their models_pytorch/expert_pytorch.py (DINOExpertModel: Meta DINOv3 + HF Gemma).
  pps_official.proxy_pytorch         their models_pytorch/proxy_pytorch.py (ProxyPytorch).

Config. ProxyPytorch only reads config.action_expert_variant, dino_model_name, dtype, freeze_dino_encoder,
action_horizon and action_dim, so ProxyCfg below carries those, with the values of their released task config
(proxy_isaaclab_droid_*_pi05_jointpos: gemma_12m, facebook/dinov3-vits16-pretrain-lvd1689m, float32) and LIBERO's
horizon and real action dim.

Real dims. Their ProxyPytorch.__init__ hardcodes ACTION_DIM = 8 ("for droid setup") for action_in_proj,
action_out_proj and state_proj, while forward_distill / forward slice actions with config.action_dim. For DROID both
are 8. For LIBERO the real action dim is 7 and the real state dim is 8 (read from the pi05_libero norm stats), so
build_proxy() rebuilds exactly those three layers with the LIBERO sizes after their __init__: action_in_proj
Linear(7, 384), action_out_proj Linear(384, 7), state_proj Linear(8, 384), initialized as their __init__ does
(nn.Linear default init). Nothing else in the module depends on ACTION_DIM.

Side effects of their __init__ that are handled here
  - torch.set_float32_matmul_precision("high") (global TF32). Their training and eval both run under it. Our pi0.5
    eval path must stay bit identical, so callers that also run pi0.5 save and restore the global precision around
    proxy calls (see precision()).
  - torch.compile wrap of sample_actions: disabled by OPENPI_DISABLE_TORCH_COMPILE=1, which this module sets. That
    method is never called here (eval uses their eval_steering.py expert call, copied in faithful_steer.py).
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import importlib.util
import os
import pathlib
import subprocess
import sys

os.environ.setdefault("OPENPI_DISABLE_TORCH_COMPILE", "1")

import torch  # noqa: E402
from torch import nn  # noqa: E402

ROOT = pathlib.Path("/workspace/external/pps_official")
COMMIT = "a22f9ad"
SRC = ROOT / "openpi/src/openpi"
FILES = {
    "pps_official.gemma": SRC / "models/gemma.py",
    "pps_official.preprocessing_pytorch": SRC / "models_pytorch/preprocessing_pytorch.py",
    "pps_official.expert_pytorch": SRC / "models_pytorch/expert_pytorch.py",
    "pps_official.proxy_pytorch": SRC / "models_pytorch/proxy_pytorch.py",
}
REWRITES = {
    "import openpi.models.gemma as _gemma": "import pps_official.gemma as _gemma",
    "from openpi.models_pytorch.expert_pytorch import DINOExpertModel":
        "from pps_official.expert_pytorch import DINOExpertModel",
    "import openpi.models_pytorch.preprocessing_pytorch as _preprocessing":
        "import pps_official.preprocessing_pytorch as _preprocessing",
}
SOURCES: dict[str, dict] = {}

# Their released task config (metadata.pt of proxy_isaaclab_droid_{pot,weight}_pi05_jointpos task/24000).
ACTION_EXPERT_VARIANT = "gemma_12m"
DINO_MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m"


def _check_commit():
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    assert head.startswith(COMMIT), f"official repo at {head}, expected {COMMIT}"


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    path = FILES[name]
    src = path.read_text()
    n_rewrites = 0
    for a, b in REWRITES.items():
        if a in src:
            src = src.replace(a, b)
            n_rewrites += 1
    SOURCES[name] = {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "import_lines_redirected": n_rewrites}
    if "pps_official" not in sys.modules:
        pkg = importlib.util.module_from_spec(importlib.machinery.ModuleSpec("pps_official", None, is_package=True))
        pkg.__path__ = []
        sys.modules["pps_official"] = pkg
    spec = importlib.util.spec_from_loader(name, loader=None, origin=str(path))
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = str(path)  # expert_pytorch resolves the dinov3 dir relative to its own __file__
    sys.modules[name] = mod
    exec(compile(src, str(path), "exec"), mod.__dict__)
    return mod


def modules():
    _check_commit()
    for n in ("pps_official.gemma", "pps_official.preprocessing_pytorch", "pps_official.expert_pytorch",
              "pps_official.proxy_pytorch"):
        _load(n)
    return sys.modules["pps_official.proxy_pytorch"]


@dataclasses.dataclass(frozen=True)
class ProxyCfg:
    action_dim: int  # real action dims (LIBERO 7)
    state_dim: int  # real state dims (LIBERO 8)
    action_horizon: int = 10  # pi05_libero
    action_expert_variant: str = ACTION_EXPERT_VARIANT
    dino_model_name: str = DINO_MODEL_NAME
    dtype: str = "float32"
    freeze_dino_encoder: bool = False


def build_proxy(cfg: ProxyCfg):
    """Their ProxyPytorch(config), with the three ACTION_DIM-sized layers resized to LIBERO's real dims."""
    pp = modules()
    prec = torch.get_float32_matmul_precision()
    net = pp.ProxyPytorch(cfg)
    torch.set_float32_matmul_precision(prec)  # callers choose; see module doc
    width = net.action_in_proj.out_features
    net.action_in_proj = nn.Linear(cfg.action_dim, width)
    net.action_out_proj = nn.Linear(width, cfg.action_dim)
    net.state_proj = nn.Linear(cfg.state_dim, width)
    return net


@contextlib.contextmanager
def precision(p: str):
    """Temporarily set torch's global float32 matmul precision (their proxies run under "high" = TF32)."""
    old = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(p)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(old)


def save_proxy(path, net, cfg: ProxyCfg, **meta):
    torch.save({"cfg": dataclasses.asdict(cfg), "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                "official_sources": SOURCES, "meta": meta}, path)


def load_proxy(path, device="cuda"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ProxyCfg(**ck["cfg"])
    net = build_proxy(cfg)
    net.load_state_dict(ck["state_dict"], strict=True)
    return net.to(device).eval(), cfg, ck.get("meta", {})
