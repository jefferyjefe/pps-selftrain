"""Paths, device selection, seeding and small IO helpers."""

import json
import pathlib
import random

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints" / "toy"
RESULTS_DIR = ROOT / "results"

# Disjoint env seed ranges so that demo collection, validation and final evaluation
# never share a starting configuration.
SEEDS = {
    "full_eval": 1_000_000,  # base on the original task
    "variant_val": 2_000_000,  # make_task.py check of base on the variant
    "variant_eval": 3_000_000,  # eval.py final numbers
    "steer_debug": 4_000_000,  # steer.py ad hoc runs
}


def get_device(name: str | None = None) -> torch.device:
    """MPS when available, else CPU. PYTORCH_ENABLE_MPS_FALLBACK is set in toy/__init__."""
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_json(obj, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def load_json(path: pathlib.Path):
    return json.loads(path.read_text())
