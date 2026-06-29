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

## Full DINO-CASF — ⏳ pending (after integration, tasks #5/#6)
- Will extend the probe to feed a fixed sketch tensor and verify CDN-noise sampling + query selection are
  seeded and bit-identical across launches.
