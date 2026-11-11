"""Convert timm's DINOv3 ViT-S/16 LVD-1689M weights into Meta's dinov3 state dict layout for the official PPS code.

The official proxy (github.com/TritiumR/pps @ a22f9ad, expert_pytorch.DINOExpertModel) builds the encoder with Meta's
own dinov3 code, torch.hub.load(<repo>/openpi/dinov3, "dinov3_vits16", source="local", weights=<repo>/openpi/dinov3/
checkpoints/dinov3_vits16.pth). That .pth is Meta's gated download and is not in the repo. timm's
vit_small_patch16_dinov3.lvd1689m is an ungated mirror of the same weights, converted by timm's eva.checkpoint_filter_fn.
This script inverts that filter (qkv.bias, absent in timm's ViT-S because the released biases are zero, is set to zero; gamma_1/2 -> ls1/2.gamma;
reg_token -> storage_tokens), loads it into Meta's module strictly, and checks the two implementations agree on
patch tokens in eval mode. Writes /workspace/checkpoints/dinov3_meta/dinov3_vits16.pth and symlinks it where the
official code looks for it.

Usage: OPENPI_DISABLE_TORCH_COMPILE=1 /workspace/openpi/.venv/bin/python scripts/faithful/convert_dino.py
"""
import json
import os
import pathlib
import sys

import torch

sys.path.insert(0, "/workspace/.cache/proxy_pylib")
import timm  # noqa: E402

DINO_DIR = pathlib.Path("/workspace/external/pps_official/openpi/dinov3")
OUT = pathlib.Path("/workspace/checkpoints/dinov3_meta/dinov3_vits16.pth")


def main():
    tm = timm.create_model("vit_small_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0).eval()
    ts = tm.state_dict()
    meta = torch.hub.load(str(DINO_DIR), "dinov3_vits16", source="local", pretrained=False).eval()
    ms = meta.state_dict()
    out = {}
    for k, v in ts.items():
        if k.endswith("attn.q_bias"):
            vb = ts[k.replace("q_bias", "v_bias")]
            out[k.replace("q_bias", "qkv.bias")] = torch.cat([v, torch.zeros_like(v), vb])
            continue
        if k.endswith("attn.v_bias"):
            continue
        k2 = k.replace("gamma_1", "ls1.gamma").replace("gamma_2", "ls2.gamma").replace("reg_token", "storage_tokens")
        out[k2] = v
    # ViT-S/16 is a distilled model whose released qkv biases are all zero, so timm builds it without qkv bias
    # (see the comment in timm's checkpoint_filter_fn). Meta's module has the bias parameter: it is zero.
    for k in ms:
        if k.endswith("attn.qkv.bias") and k not in out:
            out[k] = torch.zeros_like(ms[k])
    # keys Meta has that timm drops: rope periods (fixed buffer), qkv bias_mask (fixed), mask_token (pretrain only)
    missing = sorted(set(ms) - set(out))
    unexpected = sorted(set(out) - set(ms))
    for k in missing:
        assert k.endswith((".periods", ".bias_mask", "mask_token")), k
        out[k] = ms[k].clone()
    assert not unexpected, unexpected
    for k in out:
        assert out[k].shape == ms[k].shape, (k, out[k].shape, ms[k].shape)
    meta.load_state_dict(out, strict=True)
    torch.manual_seed(0)
    x = torch.rand(4, 3, 224, 224)
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    x = (x - mean) / std
    with torch.no_grad():
        a = tm.forward_features(x)[:, 1 + 4:]  # drop CLS and 4 registers
        b = meta(x, patch_embedding=True)
    diff = (a - b).abs().max().item()
    rel = ((a - b).norm() / a.norm()).item()
    print(json.dumps({"n_keys": len(out), "filled_from_meta_init": missing, "patch_tokens_shape": list(b.shape),
                      "max_abs_diff_vs_timm": diff, "rel_l2_diff_vs_timm": rel}, indent=1))
    assert b.shape == (4, 196, 384) and rel < 1e-4, rel
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, OUT)
    link = DINO_DIR / "checkpoints" / "dinov3_vits16.pth"
    link.parent.mkdir(exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    os.symlink(OUT, link)
    (OUT.parent / "convert_check.json").write_text(json.dumps(
        {"source": "timm/vit_small_patch16_dinov3.lvd1689m", "missing_filled": missing, "max_abs_diff": diff,
         "rel_l2_diff": rel}, indent=1))
    print("wrote", OUT, "and linked", link)


if __name__ == "__main__":
    main()
