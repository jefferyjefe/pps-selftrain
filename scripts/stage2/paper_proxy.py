"""Paper style proxy policy for pi05_libero (PPS, arXiv 2609.09148v3, Sec. 4.3 and Appendix A.1 to A.3).

About 33M trainable parameters. A DINOv3 ViT-S/16 visual encoder (21.6M, pretrained LVD-1689M weights, trained
jointly, not frozen) and a Gemma style action expert transformer (about 12M) that fuses the noised action with a
sinusoidal time embedding by concatenation and an MLP and regresses the flow matching velocity.

Choices the paper leaves open, made here.
- One encoder shared by both cameras (agentview, wrist), plus a learned camera embedding. Images are the 224 x 224
  uint8 frames pi0.5 sees (rotated and padded by pps_real.libero_obs), normalized with ImageNet mean and std.
- All 201 output tokens per camera (CLS, 4 registers, 196 patches, after DINOv3's final norm) feed the expert.
- The expert reads image tokens the way pi0's action expert reads the VLM prefix. In every layer the 10 action tokens
  attend jointly to their own keys and values and to keys and values projected from the image tokens by that layer.
  The image tokens are encoded once per observation and reused for all 10 denoising steps.
- Gemma style blocks. Pre RMSNorm (Gemma's 1 + weight form), multi query attention (8 query heads of 64, 1 KV head),
  GeGLU MLP (tanh GELU), residual stream width 512, 5 layers, final RMSNorm, a linear head to 32 action dims.
  Action tokens are bidirectional (as pi0's action chunk) with a learned position embedding.
- No language input. The paper drops it for the single task setting (Sec. 4.2).
- Time embedding is toy.model.sinusoidal (the repo's, also used by the feature based reference) on openpi's t.

Needs timm >= 1.0.20 for DINOv3. It is installed with --no-deps in /workspace/.cache/proxy_pylib so the openpi venv
stays untouched, and the weights come from timm/vit_small_patch16_dinov3.lvd1689m (ungated mirror of Meta's weights).
"""

import pathlib
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

PYLIB = "/workspace/.cache/proxy_pylib"
if PYLIB not in sys.path:
    sys.path.insert(0, PYLIB)
import timm  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from toy.model import sinusoidal  # noqa: E402

DINO_NAME = "vit_small_patch16_dinov3.lvd1689m"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * (1.0 + self.weight.float())).type_as(x)


class ExpertLayer(nn.Module):
    def __init__(self, width, heads, head_dim, kv_heads, mlp_hidden):
        super().__init__()
        self.heads, self.head_dim, self.kv_heads = heads, head_dim, kv_heads
        self.attn_norm = RMSNorm(width)
        self.q = nn.Linear(width, heads * head_dim, bias=False)
        self.kv = nn.Linear(width, 2 * kv_heads * head_dim, bias=False)
        self.img_norm = RMSNorm(width)
        self.img_kv = nn.Linear(width, 2 * kv_heads * head_dim, bias=False)
        self.o = nn.Linear(heads * head_dim, width, bias=False)
        self.mlp_norm = RMSNorm(width)
        self.gate_up = nn.Linear(width, 2 * mlp_hidden, bias=False)
        self.down = nn.Linear(mlp_hidden, width, bias=False)

    def image_kv(self, img):
        """img (N, L, W) -> k, v each (N, kv_heads, L, head_dim). Once per observation."""
        N, L, _ = img.shape
        k, v = self.img_kv(self.img_norm(img)).view(N, L, 2, self.kv_heads, self.head_dim).unbind(2)
        return k.transpose(1, 2), v.transpose(1, 2)

    def forward(self, h, img_k, img_v, rep):
        """h (B, T, W) action tokens, img_k/v (N, kv, L, hd) with B = N * rep (rep denoising states per obs)."""
        B, T, _ = h.shape
        x = self.attn_norm(h)
        q = self.q(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(x).view(B, T, 2, self.kv_heads, self.head_dim).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        if rep > 1:
            img_k = img_k.repeat_interleave(rep, 0)
            img_v = img_v.repeat_interleave(rep, 0)
        k = torch.cat([img_k, k], 2)
        v = torch.cat([img_v, v], 2)
        a = F.scaled_dot_product_attention(q, k, v, enable_gqa=self.kv_heads != self.heads)
        h = h + self.o(a.transpose(1, 2).reshape(B, T, -1))
        g, u = self.gate_up(self.mlp_norm(h)).chunk(2, -1)
        return h + self.down(F.gelu(g, approximate="tanh") * u)


class PaperProxy(nn.Module):
    def __init__(self, width=512, depth=5, heads=8, head_dim=64, kv_heads=1, mlp_hidden=1024, n_cams=2,
                 horizon=10, act_dim=32, pretrained=True):
        super().__init__()
        self.config = dict(width=width, depth=depth, heads=heads, head_dim=head_dim, kv_heads=kv_heads,
                           mlp_hidden=mlp_hidden, n_cams=n_cams, horizon=horizon, act_dim=act_dim)
        self.width, self.horizon, self.act_dim = width, horizon, act_dim
        self.encoder = timm.create_model(DINO_NAME, pretrained=pretrained, num_classes=0)
        enc_dim = self.encoder.embed_dim
        self.register_buffer("img_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1) * 255, persistent=False)
        self.register_buffer("img_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1) * 255, persistent=False)
        self.cam_emb = nn.Parameter(torch.zeros(n_cams, 1, enc_dim))
        self.img_proj = nn.Linear(enc_dim, width)
        self.action_in = nn.Linear(act_dim, width)
        self.action_time_mlp = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, width))
        self.pos_emb = nn.Parameter(torch.zeros(1, horizon, width))
        self.layers = nn.ModuleList([ExpertLayer(width, heads, head_dim, kv_heads, mlp_hidden) for _ in range(depth)])
        self.final_norm = RMSNorm(width)
        self.action_out = nn.Linear(width, act_dim)
        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.normal_(self.cam_emb, std=0.02)

    def encode(self, images):
        """images (N, n_cams, 224, 224, 3) uint8 -> image tokens (N, n_cams * 201, width). Once per observation."""
        N, C = images.shape[:2]
        x = images.flatten(0, 1).permute(0, 3, 1, 2).float()
        x = (x - self.img_mean) / self.img_std
        tok = self.encoder.forward_features(x)  # (N * C, 201, enc_dim), after the final norm
        tok = tok.view(N, C, tok.shape[1], -1) + self.cam_emb[None]
        return self.img_proj(tok.flatten(1, 2))

    def image_kv(self, tokens):
        return [layer.image_kv(tokens) for layer in self.layers]

    def velocity(self, kv, x_t, t, rep=1):
        """kv from image_kv for N observations, x_t (N * rep, 10, 32), t (N * rep,). Returns v (N * rep, 10, 32)."""
        B = x_t.shape[0]
        temb = sinusoidal(t.float(), self.width).to(x_t.dtype)[:, None].expand(B, self.horizon, self.width)
        h = self.action_time_mlp(torch.cat([self.action_in(x_t), temb], -1)) + self.pos_emb
        for layer, (k, v) in zip(self.layers, kv):
            h = layer(h, k, v, rep)
        return self.action_out(self.final_norm(h))

    def forward(self, images, x_t, t):
        """One velocity per observation. images (B, 2, 224, 224, 3), x_t (B, 10, 32), t (B,)."""
        return self.velocity(self.image_kv(self.encode(images)), x_t, t)


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def save_proxy(path, net, **meta):
    torch.save({"config": net.config, "encoder_name": DINO_NAME,
                "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "meta": meta}, path)


def load_proxy(path, device="cuda"):
    ck = torch.load(path, map_location="cpu")
    net = PaperProxy(**ck["config"], pretrained=False)
    net.load_state_dict(ck["state_dict"])
    return net.to(device).eval()
