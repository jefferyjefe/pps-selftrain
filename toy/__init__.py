"""Toy Proxy Policy Steering on PushT (state observations, flow matching)."""

import os

# Must be set before torch is imported anywhere, so ops missing on MPS run on CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
