"""
CASF sketch-conditioning modules, ported faithfully from the Deformable-DETR reference
(clip_ddetr_ow_repr/models/deformable_transformer.py:504-619) and adapted for DINO-DETR.

Load-bearing parts ported (see PLAN.md §2 / design_log/DESIGN_LOG.md):
  - RMSNorm, DePathAttn (MQA cross-attn), TinyFFN (GEGLU), DePathBlock  -> verbatim from reference.
  - SketchAlignAdapter (768->256, zero-init residual)                   -> verbatim from reference.
DINO-specific additions (DESIGN_LOG D2/D3/D5):
  - SketchMeanPool       : mean-pool of CLIP tokens (replaces inert content-adaptive token_scorer).
  - BoxSketchAdapter     : zero-init additive injection into the native 256-in LFT box head (D2).
  - GatedConcatClassHead : reference-style late gated-concat into the (binary, reinit) class head (D3).

DROPPED (verified inert, see PLAN.md §2): token_scorer/content-adaptive pooling, per-layer delta_gain
schedule, domain_embed.
"""
import math
from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------------------------------
# Verbatim from reference (clip_ddetr_ow_repr/models/deformable_transformer.py:504-599)
# ----------------------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        norm = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).rsqrt()
        return self.weight * x * norm


class DePathAttn(nn.Module):
    """Lightweight cross-attn (MQA): Q from decoder queries (d_model), shared K,V from CLIP tokens."""
    def __init__(self, d_model=256, d_clip=768, n_heads=4, head_dim=64, p_drop=0.0, share_kv=False):
        super().__init__()
        self.d_model = d_model
        self.nh = n_heads
        self.hd = head_dim
        self.do = nn.Dropout(p_drop)
        self.q_norm = RMSNorm(d_model)
        self.kv_norm = RMSNorm(d_clip)
        self.w_q = nn.Linear(d_model, n_heads * head_dim, bias=True)
        self.w_k = nn.Linear(d_clip, head_dim, bias=False)
        self.w_v = self.w_k if share_kv else nn.Linear(d_clip, head_dim, bias=False)
        self.w_o = nn.Linear(n_heads * head_dim, d_model, bias=True)
        self.log_temp = nn.Parameter(torch.zeros(1))

    def forward(self, q_BQD, kv_BTD):
        B, Q, _ = q_BQD.shape
        q = self.q_norm(q_BQD)
        kv = self.kv_norm(kv_BTD)
        qh = self.w_q(q).view(B, Q, self.nh, self.hd).transpose(1, 2)          # [B, nh, Q, hd]
        k = self.w_k(kv)                                                        # [B, T, hd]
        v = self.w_v(kv)                                                        # [B, T, hd]
        scale = torch.exp(self.log_temp) / sqrt(self.hd)
        attn = torch.einsum('b h q d, b t d -> b h q t', qh, k) * scale
        attn = attn.softmax(dim=-1)
        ctx = torch.einsum('b h q t, b t d -> b h q d', attn, v)
        ctx = ctx.transpose(1, 2).contiguous().view(B, Q, self.nh * self.hd)
        return self.w_o(self.do(ctx))


class TinyFFN(nn.Module):
    def __init__(self, d_model=256, d_ff=384, p_drop=0.0):
        super().__init__()
        self.n = RMSNorm(d_model)
        self.v = nn.Linear(d_model, d_ff)
        self.g = nn.Linear(d_model, d_ff)
        self.o = nn.Linear(d_ff, d_model)
        self.do = nn.Dropout(p_drop)

    def forward(self, x):
        x_n = self.n(x)
        h = self.v(x_n) * F.gelu(self.g(x_n))
        return self.o(self.do(h))


class DePathBlock(nn.Module):
    """Per-decoder-layer cross-attn refinement of decoder queries against CLIP sketch tokens.
    Contains its own residual: returns refined queries (NOT a delta).

    DINO-specific warm-start (DESIGN_LOG D10 addendum): with zero_init_residual=True the output
    projections (attn.w_o, ffn.o) are zero-initialised so the block is the IDENTITY at init. This
    protects DINO's strong pretrained init / delicate LFT (handover §3.2 collapse warning): the whole
    sketch conditioning starts at zero and is learned up, analogous to the reference's zero-init
    SketchAlignAdapter.up + small sk_gate. (Reference DDETR did not zero-init w_o; this is a deliberate
    deviation for DINO, logged for deliverable #4.)"""
    def __init__(self, d_model=256, d_clip=768, n_heads=4, head_dim=64, p_drop=0.0, share_kv=True,
                 zero_init_residual=False):
        super().__init__()
        self.attn = DePathAttn(d_model, d_clip, n_heads, head_dim, p_drop, share_kv)
        self.ffn = TinyFFN(d_model, d_ff=384, p_drop=p_drop)
        self.do = nn.Dropout(p_drop)
        if zero_init_residual:
            nn.init.zeros_(self.attn.w_o.weight); nn.init.zeros_(self.attn.w_o.bias)
            nn.init.zeros_(self.ffn.o.weight); nn.init.zeros_(self.ffn.o.bias)

    def forward(self, q_BQD, kv_BTD):
        q = q_BQD + self.do(self.attn(q_BQD, kv_BTD))
        q = q + self.do(self.ffn(q))
        return q


class SketchAlignAdapter(nn.Module):
    """768->256 with zero-init residual (reference)."""
    def __init__(self, in_dim=768, out_dim=256, bottleneck=96, p_drop=0.0):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.skip = nn.Linear(in_dim, out_dim, bias=False)
        self.down = nn.Linear(in_dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, out_dim)
        self.drop = nn.Dropout(p_drop)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        x_n = self.ln(x)
        return self.skip(x_n) + self.drop(self.up(self.act(self.down(x_n))))


# ----------------------------------------------------------------------------------------------------
# DINO-specific additions (DESIGN_LOG D2/D3/D5)
# ----------------------------------------------------------------------------------------------------
class SketchMeanPool(nn.Module):
    """D5: mean-pool CLIP sketch tokens -> 256-d pooled vector (drops inert content-adaptive scorer)."""
    def __init__(self, in_dim=768, out_dim=256):
        super().__init__()
        self.align = SketchAlignAdapter(in_dim, out_dim)

    def forward(self, sketch_tokens_BTD):                  # [B, T, 768]
        pooled = sketch_tokens_BTD.mean(dim=1)             # [B, 768]
        return self.align(pooled)                          # [B, 256]


class BoxSketchAdapter(nn.Module):
    """D2: additive, zero-init secondary injection into the NATIVE 256-in LFT box head.
    At init returns 0 -> box head sees the pretrained `output` unchanged (no LFT collapse)."""
    def __init__(self, d_model=256, init_gate=0.0):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.gate = nn.Parameter(torch.tensor(float(init_gate)))

    def forward(self, output_BQD, pooled_sketch_BD):       # [B,Q,256], [B,256]
        return output_BQD + self.gate * self.proj(pooled_sketch_BD).unsqueeze(1)


class GatedConcatClassHead(nn.Module):
    """D3: reference-style late gated-concat into the binary (reinit) class head.
    class_logits = Linear512( LN( cat([gate*pooled_sketch, hs], -1) ) )."""
    def __init__(self, d_model=256, num_classes=2, init_gate=0.1):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(float(init_gate)))
        self.norm = nn.LayerNorm(d_model * 2)
        self.cls = nn.Linear(d_model * 2, num_classes)
        # focal-bias init (matches DINO dino.py:137)
        prior = 0.01
        nn.init.constant_(self.cls.bias, -math.log((1 - prior) / prior))

    def forward(self, hs_BQD, pooled_sketch_BD):           # [B,Q,256], [B,256]
        sk = (self.gate * pooled_sketch_BD).unsqueeze(1).expand(-1, hs_BQD.shape[1], -1)
        decoded = self.norm(torch.cat([sk, hs_BQD], dim=-1))
        return self.cls(decoded)
