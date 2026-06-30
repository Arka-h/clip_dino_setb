# PLAN — Port CASF sketch-conditioning from Deformable-DETR to DINO-DETR

> Quarantined / background / bonus experiment. Enters the thesis ONLY if all four gates (§7) pass.
> A non-reproducible DINO number is worse than no number. When gates and speed conflict, gates win.
> If Gate 1 or Gate 2 fails → STOP and report.

## 0. Purpose
Sketch-conditioned open-world object detection ("CASF"), currently on Deformable-DETR, ported onto
**DINO-DETR** (IDEA-Research: contrastive denoising, mixed query selection, look-forward-twice) to show
the idea transfers across detector architectures. Bonus result for a future paper, not a thesis
deliverable. Correctness + reproducibility dominate speed.

## 1. Repository discipline (NON-NEGOTIABLE)
Sources of truth — ONLY these three (read, do not copy bugs in):
- `clip_ddetr_ow_repr`     → reference CASF impl + headline checkpoint/recipe (conditioning mechanism)
- `clip_ddetr_analysis`    → deterministic eval + ablation harness (determinism, seed-14, pycocotools 12-stat, size strata)
- `clip_ddetr_clean_run`   → clean training recipe (cos34 schedule, optimizer/lr, resume, data pipeline)

DO NOT pull code/weights/configs from: `clip_ddetr` (legacy), `clip_ddetr_base`, `gdino_clip`,
`locformer-SGOL`, `sketch_detr`, `SLIP`, `clip_dino`. Especially do NOT lift DINO/GroundingDINO from
`gdino_clip`. Start DINO from clean upstream IDEA-Research DINO (COCO-pretrained ResNet-50).
If something needed is not in the three allowed repos → flag and ask.

## 2. CASF parts to PORT (load-bearing) vs DROP (inert)
PORT:
1. Frozen CLIP-ViT sketch encoder + sketch-token formation (CLS + patch tokens, ~50 tokens @ 768-d). Frozen.
2. `DePathBlock` — per-layer cross-attn: Q=decoder queries (256), K/V=CLIP sketch tokens (768, shared),
   4 heads, residual + tiny FFN, learned temperature. Refines queries BEFORE each decoder layer. PRIMARY (−0.145 AP ablation).
3. Late gated-concat into box head: sk = sk_gate[lid]·pooled_sketch; cat([sk, output]); norm; bbox_head. SECONDARY (−0.127 AP). sk_gate init 0.1.
4. Determinism + eval + training protocol (§4–5).
DROP (verified inert):
- Content-adaptive token pooling (token_scorer) → use plain MEAN-POOL for pooled_sketch.
- Per-layer delta_gain schedule → use DINO native refinement; at most a single constant scalar.
- domain_embed (14 prompt tokens) → dead/unregistered in reference; DO NOT port. CLS+patches only.

## 3. DINO landmines (document every decision → Gate 3)
1. CDN groups: does DePathBlock apply to denoising queries too? Default: condition ALL queries uniformly;
   verify CDN attn mask not broken; CDN is train-only → confirm no train/eval mismatch.
2. Look-Forward-Twice box refinement: how sketch-conditioned feat enters bbox_embed; do NOT transplant
   delta_gain blindly (forcing gain=1.0 collapsed reference AP to ~0.001). Constant scalar at most, or none.
3. Mixed query selection: confirm per-layer conditioning composes with encoder-initialised queries.
4. Dynamic anchor boxes (4-D): keep DINO native anchor update; do not import DDETR reference-point logic.
5. Two-stage encoder proposals: sketch signal must NOT leak into encoder proposals (decoder-only conditioning).
6. Determinism re-verification (HARD): re-verify bit-identical on DINO incl. CDN noise + query selection.

## 4. Eval protocol (copy from clip_ddetr_analysis) — EXACT
- Determinism (set BEFORE building model): cudnn.deterministic=True, cudnn.benchmark=False,
  torch.use_deterministic_algorithms(True), CUBLAS_WORKSPACE_CONFIG=:4096:8. Two launches → bit-identical AP (max abs diff 0.0).
- Query sampling: seed-14, one category per image, binary/class-agnostic GT, single query (load-1/use-1).
- Metric: pycocotools full 12-stat. Report for every cell, OW (held-out 14) and CW (all-56).
- Two eval modes from ONE model: OW (open_qd, held-out 14) and CW (closed_qd, all-56).
- Retention = OW AP ÷ CW-holdout-14 AP (mAP, AP@50, macro).

## 5. Training recipe (copy from clip_ddetr_clean_run)
- Base: clean upstream DINO-DETR R50 COCO-pretrained (resume from official COCO DINO-R50 ckpt).
  Freeze CLIP-ViT sketch encoder; freeze image backbone per reference regime.
- Schedule cos34: warmup 1e-5→8e-5 over 5 ep · cosine k=1 → 7.5e-6 ending ep 34 · hold. AdamW lr 1e-4 /
  backbone 2e-5, effective batch 8 (2-GPU or grad-accum, never silently batch 4), 50 epochs.
- Loss: binary (object/no-object) set prediction (DINO Hungarian + L1 + GIoU, binarised), consistent across CDN + matching.
- Data: MS-COCO 2017 + QuickDraw, 56 COCO-intersecting categories. query=(image, sketch of one cat); target=all instances.
- Convention: train OW → get CW for free. One training run; eval both. NO separate CW training.
- Holdout Set B (14, hold OUT, eval on): backpack, bicycle, clock, couch, dog, elephant, knife, mouse,
  oven, pizza, sandwich, skateboard, stop sign, train. Train on remaining 42. Log holdout with run.

## 6. Order of work
1. Stand up clean DINO-R50 (COCO-pretrained); confirm runs + deterministic on plain detection smoke (Gate 1 on base).
2. Port CLIP sketch encoder + token formation (frozen; CLS+patches; no domain_embed).
3. Port DePathBlock (per-layer), resolve CDN interaction (§3.1) → log decision.
4. Port late gated-concat into DINO box head, resolve LFT (§3.2) → log. Mean-pool pooled_sketch.
5. Re-verify determinism on full DINO-CASF (Gate 1), incl. CDN/query-select seeding.
6. Short sanity train (few epochs) → CW/OW sane + improving (Gate 2) BEFORE full run.
7. Full cos34 run (Set B held out) → OW + CW 12-stat. Log everything (§8).
STOP and report if Gate 1 or Gate 2 fails.

## 7. QUARANTINE GATES (hard stops)
- Gate 1 Determinism: two launches bit-identical AP (base AND full DINO-CASF), all stochastic sources seeded. Fail→STOP.
- Gate 2 Sanity: CW>OW, retention sane band, not collapsed (~0), not OW≈CW (holdout leak signal). Reproduce on short train.
- Gate 3 Design-log: every DINO decision (CDN, LFT, query-select, anchor, encoder-freeness) written + rationale. Deliverable.
- Gate 4 Comparability: identical Set B holdout, seed-14 single-query eval, pycocotools 12-stat, retention calc vs DDETR headline.

## 8. Logging / naming
- Match runs by wandb_name/commit, NEVER by mAP. Distinct wandb_name + output dir per run; never overwrite a ckpt.
- OW run provenance bundle: {held-out 14, seed, commit, wandb_name, output dir, four design-log decisions}.
- No cherry-picking: report every run launched.

## 9. Deliverables
1. DINO-CASF OW (Set B) + CW reference, deterministic 12-stat.
2. Retention (mAP/AP@50/macro), identical calc to thesis.
3. Design-log (Gate 3).
4. "what changed vs DDETR CASF, and why" note.
5. Determinism verification log (Gate 1 evidence).
6. Provenance bundle (§8).

## 10. Self-check
- OW≈CW → suspect holdout leak (Set B seen in training, or sketch leaking via encoder). Re-check §3.5 + partition.
- Determinism won't hold → unseeded stochastic source (CDN noise / query selection).
- "learning from scratch"/plateau → representation/dim mismatch at an injection point (DePathBlock dims, box-head concat, anchor).
- Asymmetry: upside = generality bonus; downside = non-reproducible number contaminating a reproducible thesis. Gates win.

---
## Working notes / status (live)
- Working dir: /home/rahul/arka/clip_dino_setb (quarantined, NOT one of the 3 source repos).
- mAP/AP reported as decimals (e.g. 0.419), never percentage points. [user pref]
- Git commits: NO trailers. [repo rule]
