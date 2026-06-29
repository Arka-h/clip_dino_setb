# DINO-CASF Design Log (Gate 3 deliverable)

Every DINO-specific decision for porting CASF (Deformable-DETR → DINO-DETR), with rationale and exact
code anchors. **No silent choices.** Status markers: ✅ decided · ⚠️ decided-but-verify-empirically ·
❓ open (needs a run or a user call).

Anchors use the clean upstream clone at `dino_casf_port/DINO_upstream` (IDEA-Research DINO, commit
`d84a491`). Reference CASF anchors are in `clip_ddetr_ow_repr` (`models/deformable_transformer.py`,
`models/deformable_detr.py`). Training/eval discipline anchors are in `clip_ddetr_clean_run` /
`clip_ddetr_analysis`.

---

## D0. Base model & checkpoint ✅
- **Code:** clean upstream DINO, config `config/DINO/DINO_4scale.py` (R50, 4 feature levels, 6 enc / 6 dec
  layers, hidden 256, FFN 2048, 8 heads, 900 queries, num_select 300, two_stage_type `standard`, CDN
  `dn_number` default 100).
- **Weights:** official COCO checkpoint `checkpoint0011_4scale.pth` (verified: `model` has 626 tensors,
  `transformer.*`, `class_embed.*` = 91-way, args.num_queries=900, backbone=resnet50). It currently lives
  under the forbidden `clip_dino/ckpts/` dir, but it is the *official IDEA-Research weights file*
  (timestamp Jul-2022, standard layout) — only the **weights** are reused, never `clip_dino` code. We will
  copy it into `dino_casf_port/` to decouple from that repo.
- **Ops:** `models/dino/ops` CUDA kernels compiled against torch 2.7/cu128 after a one-line compat patch
  (`AT_DISPATCH_FLOATING_TYPES(value.type()…)` → `value.scalar_type()`) — build-compat only, no logic
  change. `.so` built for sm_75 (Quadro RTX 8000).

---

## D1. CDN (contrastive denoising) interaction — landmine §3.1 ✅

**Decision:** the sketch cross-attention (`DePathBlock`) conditions **all** decoder queries uniformly —
both the CDN denoising-pad queries and the matching queries — and is placed so it is **orthogonal to the
CDN self-attention mask**.

**Why this is safe (traced):**
- In DINO, the decoder `tgt`/`output` is `[nq_total, bs, d]` where during training
  `nq_total = pad_size + num_queries` (CDN pad prepended): `prepare_for_cdn` builds `tgt_size = pad_size +
  num_queries` and the `attn_mask` of that size (`models/dino/dn_components.py:112-124`). That mask gates
  **query↔query self-attention only** (passed as `self_attn_mask=tgt_mask` to the decoder layer,
  `deformable_transformer.py:726`).
- `DePathBlock` is **query→sketch-token cross-attention** (Q = decoder queries, K/V = CLIP sketch tokens).
  It introduces **no query↔query interaction**, so it can neither leak across the CDN/matching boundary
  nor be corrupted by `attn_mask`. We inject it *before* `layer(...)` (i.e. before self-attn), operating
  per-query independently → the mask is untouched.
- **No train/eval mismatch:** CDN is train-only (`dino.py:262` gated on `self.training and dn_number>0`).
  At eval `pad_size=0`, so `output` is just the `num_queries` matching queries and the *same* per-query
  conditioning applies. Because conditioning is per-query and mask-independent, removing the pad rows at
  eval changes nothing about how a matching query is conditioned. ✅
- **Loss consistency:** the binary (object/no-object) objective (D7) is applied identically to CDN and
  matching groups; `dn_post_process` (`dino.py`) splits outputs back into dn/matching after the heads, and
  both go through the same binarised criterion. We do **not** special-case CDN in the heads.

**Injection point (exact):** in `TransformerDecoder.forward` (`deformable_transformer.py:676` loop), after
`query_pos` is computed (`:696`) and immediately before the `output = layer(...)` call (`:712-728`):
```
# output: [nq_total, bs, 256] ; sketch_tokens: [bs, T, 768]
output = output + DePathBlock_layer(q_BQD=output.transpose(0,1), kv_BTD=sketch_tokens).transpose(0,1)
```
(`DePathBlock` already contains the residual internally; we keep its internal residual and do NOT
double-add — shown additive here only for clarity; final code calls `output = refine_output[lid](...)`.)

---

## D2. Look-Forward-Twice (LFT) box-head integration — landmine §3.2 ✅/⚠️

**Context:** DINO refines a 4-D anchor per layer. `self.bbox_embed[layer_id](output)` produces
`delta_unsig`, added to `inverse_sigmoid(reference_points)` → next anchor (`deformable_transformer.py:731-
735`); the **same** `bbox_embed` modules are reused on `hs` for the final boxes
(`dino.py:151, 277-282`) — that sharing *is* Look-Forward-Twice. The pretrained box head is a
`MLP(256,256,4,3)` with **zero-init last layer** (`dino.py:138-139`) — delicate; the handover warns
forcing a foreign scalar/oversized head onto LFT collapsed reference AP to ~0.001.

**Decision (recommended, safe mapping):**
- **Primary conditioning already covers the box path.** Because `DePathBlock` (D1) conditions `output`
  *before* every decoder layer, `output` is already sketch-aware when it reaches `bbox_embed`. So DINO's
  native LFT refinement is sketch-conditioned **without touching the pretrained box head**.
- **Secondary signal into the box path = additive, zero-initialised adapter** (NOT concat, NOT a foreign
  delta_gain): keep `bbox_embed` at its native **256-in** (preserve pretrained weights + LFT init), and
  feed it `output_box = output + sk_gate_box[lid] · zeroinit_Linear256(pooled_sketch_256)`. `sk_gate_box`
  init **0.0** and the adapter `Linear(256,256)` **zero-initialised** ⇒ at step 0 the box head sees exactly
  the pretrained `output` (no localisation perturbation, no collapse risk), and it can *learn* to use the
  sketch. This is the DINO-faithful analogue of the reference's zero-init `SketchAlignAdapter.up`.
- **No `delta_gain`.** DINO has native LFT; we add no per-layer box-delta scalar (handover: "a constant
  scalar at most, or none" → we choose **none**). ✅

**Deviation from reference (logged for deliverable #4):** the reference concatenates the gated sketch into
a **512-in** box head (reinitialising its first layer). On DINO we deliberately do **not** widen/reinit
the box head, to protect the pretrained LFT regressor; we deliver the same secondary intent additively with
a zero-init gate. ⚠️ If Gate-2 sanity shows the box path under-uses the sketch, the fallback is the
reference-style 512-in concat box head (reinit) — but only if it does not collapse localisation.

---

## D3. Late gated-concat into the CLASS head ✅

**Decision:** apply the reference-style late gated-concat to the **binary class head**, which is
reinitialised anyway (D7), so widening it is free.
- Form `decoded_feat = LayerNorm_512( concat([ sk_gate_cls[lid]·pooled_sketch_256 , hs_layer ], dim=-1) )`
  and set `class_embed = Linear(512, num_classes)` (vs DINO native `Linear(256, num_classes)`,
  `dino.py:132`). `sk_gate_cls` init **0.1** (matches reference `sk_gate`).
- Applied at both head sites that exist in DINO: the in-decoder `class_embed[layer_id](output)` used for
  query selection (`deformable_transformer.py:742`) **and** the final `class_embed` over `hs`
  (`dino.py:284-285`). Same module (shared), so one widened head covers both. ✅
- `pooled_sketch` = **mean-pool** of the sketch tokens (D5), projected 768→256 by a small adapter
  (LayerNorm+Linear), matching the reference's `sketch_align` role but with mean-pool not token-scorer.

**Rationale:** keeps the secondary signal exactly where the reference put weight on it (the head), but on
the head we are free to reinit (binary), so we incur no pretrained-weight loss and avoid the box-head
collapse risk (handled separately in D2).

---

## D4. Mixed query selection composition — landmine §3.3 ✅
- DINO initialises decoder content+anchor queries from the encoder top-k (`deformable_transformer.py:330-
  376`, two_stage `standard`). CASF conditioning is **downstream** of selection (inside the decoder loop,
  D1) and operates on whatever `tgt`/`refpoint` the selection produced. Dims match (`output` is `[nq,bs,
  256]` regardless of how queries were initialised). **No change to query selection.** The encoder/
  selection path never sees the sketch (D6). ✅

## D5. Pooled-sketch formation ✅ (DROP content-adaptive)
- `pooled_sketch = mean over sketch tokens` (CLS+patch, ~50 tokens, 768-d) → `Linear/LN 768→256`.
  Per handover §2: content-adaptive `token_scorer` is verified inert (−0.0006 AP); we use plain mean-pool.
- DePathBlock K/V uses the **full token set** (not the pool); the pool is only for the late concat (D3) and
  box adapter (D2). ✅

## D6. Encoder sketch-freeness (holdout semantics) — landmine §3.5 ✅
- The sketch signal enters **only** in the decoder (D1–D3). The encoder, two-stage proposal generation
  (`enc_out_class_embed`/`enc_out_bbox_embed`, `dino.py:159-169`), and query selection are **sketch-free**.
  This preserves the meaning of "held-out": Set-B never influences proposals. Encoder-side fusion is
  explicitly out of scope. **Guard:** a unit assert that no sketch tensor reaches `transformer.encoder` /
  `enc_out_*`. ✅

## D7. Binary objective + head reinit ✅
- `num_classes = 2` (object=label 1, no-object slot), GT labels relabelled to 1 (binary/class-agnostic),
  focal-bias init on the class head (`dino.py:137`). Load COCO checkpoint with shape-filtered
  `strict=False` so the 91-way `class_embed`/`enc_out_class_embed` (and, in our model, the now-512-in class
  head) are dropped and reinitialised — same mechanism as `clip_ddetr_clean_run` `filter_ckpt`
  (`main.py:196-211`). Box head stays 256-in (D2) so its pretrained weights **do** load. Criterion: DINO
  Hungarian + L1 + GIoU + focal, binarised, consistent across CDN & matching (D1). ✅

## D8. CLIP sketch encoder + freeze regime ✅
- Frozen CLIP ViT-B/32 (width 768, 12 layers, patch 32, 224²) loaded from
  `checkpoints/clip_model/vit_clip2.pth`; sketch tokens = CLS + 49 patch = 50 tokens @ 768-d.
- **Freeze = `partial`** (match reference): all CLIP params frozen **except LayerNorm affine** (excluding
  `ln_post`) — `clip_ddetr` `build_deforamble_transformer` partial branch. Handover §5 says "freeze
  whatever the reference freezes"; reference cos34 default is `partial`. ⚠️ Note: this means CLIP is *not*
  100% frozen (LN affine trains). Logged as a deliberate match-the-reference choice; alternative `full`
  freeze is a one-flag switch if we want strict freezing.
- **DROP `domain_embed`** (dead/unregistered in reference). Sketch tokens = CLS+patches only.
- Sketch preprocessing: sketch-rnn stroke-3 → rasterize white-on-black → normalize with CLIP mean/std
  (from `clip_ddetr_clean_run` `datasets/coco.py`). ⚠️ Note: reference runs the CLIP encoder *interleaved*
  2 blocks/decoder-layer; for the port we form the **full 50-token representation once** (run all 12 CLIP
  blocks up front) and reuse it across the 6 DINO decoder layers — simpler, and equivalent for a frozen
  encoder since there is no per-layer sketch update on the 'De' path that depends on decoder state (the
  reference's per-layer `clip_vits.layers(...)` advance is just finishing the frozen forward). Logged as a
  deviation (deliverable #4).

## D9. Image backbone freeze ✅
- ResNet-50, FrozenBatchNorm, train `layer2/3/4`, freeze stem+`layer1` (DINO default with
  `lr_backbone>0`). Matches reference regime.

## D10. DePathBlock hyperparameters ✅ (copy reference exactly)
- `DePathAttn`: d_model 256, d_clip 768, n_heads 4, head_dim 64 (4·64=256), **MQA** (shared single-head
  K/V, `share_kv=True`), RMSNorm on Q (256) and K/V (768), learned `log_temp` (scale =
  exp(log_temp)/√head_dim). `DePathBlock` = cross-attn(+residual) → TinyFFN (GEGLU, d_ff 384, +residual).
  One block per decoder layer (6), built as `refine_output = ModuleList([DePathBlock(...) ×6])`. ✅
  These are *new* params (not in the COCO checkpoint) → trained from init.
- **D10 addendum — zero-init warm-start (DINO deviation, deliverable #4):** DePathBlock is built with
  `zero_init_residual=True` → `attn.w_o` and `ffn.o` are zero-initialised, so the block is the IDENTITY at
  init. Combined with BoxSketchAdapter gate=0 (D2), the *entire* sketch conditioning is identity on the
  localisation path at init, so the full DINO-CASF reproduces base DINO's boxes bit-identically at step 0
  (verified: max|Δ pred_boxes|=0.0) and the sketch influence is *learned up*. The reference DDETR did NOT
  zero-init `w_o`; we add it to protect DINO's strong pretrained init / delicate LFT (§3.2 collapse
  warning). Class head keeps `sk_gate` init 0.1 (reference value) since the binary head is reinit anyway.

---

## D11. Determinism re-verification plan — Gate 1 (landmine §3.6) ❓(needs GPU)
Re-verify bit-identical across two process launches, **including DINO's extra stochastic sources**:
1. `enable_determinism()` BEFORE `build_model` (CUBLAS_WORKSPACE_CONFIG=:4096:8, cudnn.deterministic=True,
   benchmark=False, use_deterministic_algorithms(True, warn_only=True), then manual_seed) — order matters
   (clip_ddetr_analysis `main.py:254-274`).
2. **CDN noise sampling** (`prepare_for_cdn` uses `torch.rand_like`/`randint` on cuda) — seed and verify;
   train-only, but determinism must hold for resumable training. Eval has no CDN.
3. **Query selection top-k** — deterministic given fixed inputs (`torch.topk`), but `random.random()` in
   `dec_layer_dropout`/`decoder_query_perturber` must be off/seeded at eval (defaults off).
4. **seed-14 eval sampling** with `n_sketches=1` (single-query; reference default is 3 — we override to 1
   per handover §4). Fresh `random.Random(14)` per image for cat pick and sketch pick.
5. Probe: two separate `python` launches → capture `pred_logits`/`pred_boxes`, assert max abs diff 0.0.
   (No `probe_xproc.py` in the analysis repo — we write our own under `smoke/`.)
**Run on BOTH** base DINO (before CASF) and full DINO-CASF. ✅plan / ❓execution blocked on free GPU.

---

## Open items / flags
- ❓ **GPU contention:** at port time GPU0 (Quadro RTX 8000, 48 GB) is fully held by another user's
  process (PID 2263953, 48 GB). No GPU runs possible until it frees. CPU-side engineering proceeds.
- ❓ **Effective batch:** handover §5 says effective batch **8**; the reference *canonical* cos34 OW job
  actually ran 2-GPU × 8 = **16**, single-GPU variant = 8. With one GPU here, DINO-4scale (900 q + CDN) at
  batch 8 may not fit 48 GB. Plan: per-GPU micro-batch as large as fits + **gradient accumulation** to
  effective 8 (the clean repo has *no* grad-accum → we must add it; never silently run batch 4). Final
  micro-batch TBD by a memory probe once the GPU frees.
- ⚠️ **delta_gain absent** by design (D2) — flagged so a reviewer doesn't "look for" it.
