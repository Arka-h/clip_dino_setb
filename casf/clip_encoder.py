"""Frozen CLIP ViT-B/32 sketch encoder for DINO-CASF (DESIGN_LOG D8).

CLIP VisionTransformer ported VERBATIM from the allowed reference
clip_ddetr_ow_repr/models/deformable_transformer.py:360-499 (LayerNorm, QuickGELU,
ResidualAttentionBlock, Transformer, VisionTransformer). The SketchCLIP wrapper:
  - loads checkpoints/clip_model/vit_clip2.pth (strip 'visual.' prefix, strict=False),
  - partial-freeze (all frozen except LayerNorm affine, excluding ln_post) — matches reference cos34 default,
  - encode([B,3,224,224]) -> full 50-token representation [B,50,768] (all 12 blocks once; D8 deviation:
    not interleaved per decoder layer). NO domain_embed (dropped, D8).
"""
from collections import OrderedDict
import torch
import torch.nn as nn


class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        orig = x.dtype
        return super().forward(x.type(torch.float32)).type(orig)


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, attn_mask=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width, layers, heads, attn_mask=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution=224, patch_size=32, width=768, layers=12, heads=12, output_dim=512):
        super().__init__()
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward_tokens(self, x):
        """[B,3,224,224] -> [B,50,768] full token sequence (CLS + 49 patches), pre-ln_post."""
        x = self.conv1(x)                                   # [B, 768, 7, 7]
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)   # [B, 49, 768]
        cls = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)                      # [B, 50, 768]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)                              # NLD -> LND
        x = self.transformer(x)                             # all 12 blocks
        x = x.permute(1, 0, 2)                              # LND -> NLD
        return x                                            # [B, 50, 768]


class SketchCLIP(nn.Module):
    def __init__(self, ckpt_path, freeze='partial'):
        super().__init__()
        self.vit = VisionTransformer(224, 32, 768, 12, 12, 512)
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        sd = {k.replace('visual.', ''): v for k, v in ck.items()}
        missing, unexpected = self.vit.load_state_dict(sd, strict=False)
        print(f"[SketchCLIP] load vit_clip2: missing={len(missing)} unexpected={len(unexpected)}")
        self._apply_freeze(freeze)

    def _apply_freeze(self, freeze):
        if freeze == 'full':
            for p in self.vit.parameters():
                p.requires_grad = False
        elif freeze == 'partial':
            for p in self.vit.parameters():
                p.requires_grad = False
            for n, p in self.vit.named_parameters():
                if ('ln' in n) and ('ln_post' not in n):
                    p.requires_grad = True
        # 'none' -> all trainable
        n_train = sum(p.requires_grad for p in self.vit.parameters())
        print(f"[SketchCLIP] freeze='{freeze}': {n_train} trainable param-tensors")

    @torch.no_grad()
    def _noop(self):
        pass

    def forward(self, sketch_BCHW):                          # [B,3,224,224]
        return self.vit.forward_tokens(sketch_BCHW)         # [B,50,768]
