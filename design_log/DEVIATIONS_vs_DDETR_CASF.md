# Deliverable #4 — What changed vs the Deformable-DETR CASF, and why

The CASF *idea* (frozen-CLIP sketch tokens → per-decoder-layer cross-attention `DePathBlock` →
late gated fusion into the heads, binary open-world objective, seed-14 single-query eval) is ported
faithfully. The deviations below are forced by DINO-DETR's architecture (CDN, Look-Forward-Twice,
mixed query selection, two-stage) or are small robustness choices. Each is logged with rationale.

| # | Reference (Deformable-DETR CASF) | DINO-CASF port | Why |
|---|---|---|---|
| 1 | **Box head**: gated sketch **concatenated** → 512-in box MLP (first layer reinit) | **Native 256-in box head preserved**; sketch enters as an **additive zero-init adapter** (`BoxSketchAdapter`, gate 0) | DINO's Look-Forward-Twice box regressor is pretrained + delicate; handover §3.2 warns a foreign/oversized box head collapses localisation (ref AP→0.001). Zero-init ⇒ identity at start (verified max\|Δ pred_boxes vs base\|=0.0), pretrained LFT intact. Primary `DePathBlock` already makes `output` sketch-aware before the box head. |
| 2 | `delta_gain` per-layer box-delta scalars | **None** | DINO has native LFT refinement; handover §2/§3.2 → "a constant scalar at most, or none". We chose none (one fewer foreign knob on LFT). |
| 3 | `DePathBlock` output projections at default init | **`w_o`/`ffn.o` zero-initialised** (`zero_init_residual=True`) | Makes the whole sketch path identity at init → protects DINO's strong pretrained init across all 6 decoder layers; the conditioning is learned up from zero (analogous to ref's zero-init `SketchAlignAdapter.up`). |
| 4 | Content-adaptive token pooling (`token_scorer`, query-aware softmax, β-residual) | **Plain mean-pool** (`SketchMeanPool`) | Verified inert in the reference (−0.0006 AP); handover §2 → drop. |
| 5 | 14 learnable `domain_embed` prompt tokens | **Dropped** | Dead/unregistered in the reference (`nn.Parameter(...).cuda()` de-registers → never trained/saved). Sketch tokens = CLIP CLS+patches only. |
| 6 | CLIP encoder run **interleaved** (2 blocks per decoder layer) | CLIP run **once**, all 12 blocks → 50 tokens reused across layers | Equivalent for a frozen encoder (the per-layer advance just finishes the frozen forward); simpler and cheaper. |
| 7 | No CDN (Deformable-DETR has no contrastive denoising) | `DePathBlock` conditions **all** decoder queries (CDN pad + matching) uniformly, **orthogonal to the CDN self-attn mask** | DINO-specific. `DePathBlock` is query→sketch *cross*-attn (no query↔query), so it neither breaks nor is broken by the CDN mask; CDN is train-only ⇒ no train/eval mismatch (D1). |
| 8 | Eval default 3 sketches/query | **Single-query** (`n_sketches=1`) | Handover §4 mandates single sketch (load-1/use-1). |
| 9 | Data subset draw: inline `np.random.choice(replace=True)`, subset unseen too | **Seeded, replace=False, nested, full-unseen-exclusion** (`subset_select.py`) + versioned cache | Fixes a <100%-data Set-B leak (the old draw shrank the held-out exclusion). See `LEAKAGE_FIX.md`. |

**Unchanged (faithful):** DePathAttn MQA geometry (Q=256, K/V=768 shared, 4 heads, head_dim 64, learned
temperature, RMSNorm), TinyFFN (GEGLU d_ff 384), late gated-concat into the (binary, reinit) class head
(`GatedConcatClassHead`, 512-in, sk_gate init 0.1), partial CLIP freeze (LN-affine trainable), backbone
freeze regime, cos34 schedule, 3-group AdamW, binary set-prediction objective, seed-14 / pycocotools
12-stat / retention = OW_AP÷CW_AP eval protocol, Set B = i%4 holdout (== handover 14).
