# Gate 1 — Determinism verification log

Protocol (from clip_ddetr_analysis): determinism flags set BEFORE build_model
(`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `cudnn.deterministic=True`, `cudnn.benchmark=False`,
`use_deterministic_algorithms(True, warn_only=True)`, seed 0), then two **separate process launches** must
produce bit-identical model outputs (max abs diff 0.0).

Probe: `smoke/probe_base_dino.py` — fixed 3×800×800 input (generator seed 1234), eval mode, compares
`pred_logits` and `pred_boxes`. GPU: Quadro RTX 8000, torch 2.7+cu128, ops sm_75.

## Base DINO-4scale (official COCO checkpoint) — ✅ PASS  [2026-06-30]
- Checkpoint load: **missing=0, unexpected=0** (clean upstream DINO matches the official checkpoint exactly).
- run1: pred_logits sum=-537188.000000, pred_boxes sum=2677.728271
- run2: pred_logits sum=-537188.000000, pred_boxes sum=2677.728271
- **max|Δ pred_logits| = 0.0 ; max|Δ pred_boxes| = 0.0 → GATE 1 (base) PASS**

This establishes the deterministic foundation BEFORE any CASF is added (handover §6 step 1 / §7 Gate 1
on the base). The full DINO-CASF model must re-pass this same probe (with CDN-noise and query-selection
seeding) after integration — tracked as task #7.

## Full DINO-CASF — ✅ PASS  [2026-06-30]  (probe: smoke/probe_casf_dino.py, fixed image + fixed sketch)
Built casf=True, num_classes=91 (so enc_out_class_embed loads pretrained → identical query selection).
Checkpoint load: kept=602, dropped=24 (the now-512-in gated decoder class heads → reinit, by design D7),
missing=333 (new CASF params + separately-loaded CLIP), unexpected=0. SketchCLIP loaded vit_clip2
missing=0/unexpected=0; partial-freeze → 50 trainable LN param-tensors.

- **Zero-init box invariant:** with DePathBlock zero_init_residual=True (identity) AND BoxSketchAdapter
  gate=0 (identity), the whole sketch conditioning is identity on the localisation path at init.
  pred_boxes sum = 2677.728271 == base DINO 2677.728271. **max|Δ pred_boxes vs base DINO| = 0.0 → PASS**
  (pred_logits differ — gated binary class head is reinit — expected). Pretrained DINO LFT fully preserved.
- **Gate 1 full (EVAL, two launches):** max|Δ pred_logits| = 0.0 ; max|Δ pred_boxes| = 0.0 → PASS.
- **Gate 1 full (TRAIN, CDN active, two launches):** dn_meta populated (pad_size/num_dn_group/
  output_known_lbs_bboxes) → DePathBlock composes with CDN; max|Δ pred_logits| = 0.0 ;
  max|Δ pred_boxes| = 0.0 → PASS. Confirms CDN-noise sampling + query selection are deterministic with
  the seed set before build (no unseeded stochastic source).

Gate 1 (§7) is satisfied for BOTH base and full DINO-CASF. Note: the eval-time AP-determinism on real
data still needs re-confirmation once the dataset adapter (task #6) exists, but the model-output bit-
identity (the core of Gate 1) is established.
