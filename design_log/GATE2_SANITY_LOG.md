# Gate 2 — Sanity train + CW/OW check

**Verdict: ✅ PASS** (2026-07-01). Cleared to proceed to the full cos34 run (#9).

## Run config (sanity, NOT headline)
- `casf/train_casf.py --sched constant --data_frac 0.1 --epochs 4 --microbatch 4 --effective_batch 8`
- Constant LR 1e-4 (skip cos34 5-epoch warmup so a short run actually learns), 8731 train imgs (10%, OW
  open scheme, Set B held out), eff batch 8 (accum 2), seed 42.
- Eval: seed-14 single-query, capped at 600 imgs/mode (sanity speed; NOT full-val, so not Gate-4-comparable).
- Checkpoint: outputs/casf_ow_sanity/checkpoint.pth ; provenance.json written.

## Loss (improving, monotone) ✓
ep0 9.59 → ep1 8.86 → ep2 8.68 → ep3 8.38  (it0 was 25.3 → learning, not collapsing).

## Results (AP = decimals)
| metric | OW (open_qd / Set-B, 14) | CW (closed_qd / 56) |
|---|---|---|
| **AP** | **0.258** | **0.362** |
| AP@50 | 0.333 | 0.501 |
| AP@75 | 0.277 | 0.377 |
| AP_s / m / l | 0.034 / 0.151 / 0.533 | 0.168 / 0.350 / 0.596 |
| AR@1/10/100 | 0.249 / 0.465 / 0.635 | 0.234 / 0.544 / 0.681 |

**Retention = OW_AP / CW_AP = 0.258 / 0.362 = 0.713.**

## Gate-2 criteria check
- **CW > OW:** 0.362 > 0.258 ✓ (correct direction — better on seen cats).
- **Not collapsed (~0):** both healthy ✓.
- **Not OW≈CW (leak signal):** clear ~40% gap, retention 0.71 (not ≈1.0) → **no holdout leak** ✓. Consistent
  with the now-verified leak-free partition + sketch-free encoder (D6).
- **Retention in a sane band:** 0.713 — below the fully-trained Deformable-DETR thesis headline (0.876,
  `clip_ddetr_analysis/analysis_findings.md`) as expected for a 4-epoch/10%-data undertrained model, and far
  above a degenerate baseline (e.g. Locformer 0.254). Sane ✓.

## Caveats / open before #9
1. **Not the headline number** — constant LR, 10% data, 4 epochs, 600-img eval. The full cos34 50-epoch /
   100%-data / full seed-14 eval run (#9) produces the comparable figure.
2. **Train-time determinism:** a `MemoryEfficientAttention non-deterministic backward` warning appears at
   train time (eval determinism already proven, Gate 1). For the full run's bit-reproducible resumability,
   set `torch.use_deterministic_algorithms(True, warn_only=False)` and/or force the math/flash SDPA kernel,
   then re-confirm a two-launch train-step diff. Address before/at #9 launch.
