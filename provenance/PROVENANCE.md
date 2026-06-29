# Provenance bundle (§8) — DINO-CASF port

## Base
- Clean DINO code: IDEA-Research DINO, commit d84a491d41898b3befd8294d1cf2614661fc0953 (cloned to DINO_upstream/).
- COCO weights: dino_r50_4scale_coco_official.pth (official IDEA-Research DINO-4scale-R50 12ep; copied from
  clip_dino/ckpts/checkpoint0011_4scale.pth — WEIGHTS ONLY, no clip_dino code used). 626 tensors, class_embed 91-way.
- Ops: models/dino/ops compiled torch2.7/cu128, sm_75, with value.type()->value.scalar_type() compat patch.

## Assets
- COCO 2017: /mnt/1tb/data/coco (train2017, val2017, annotations + qd_train_{open,closed}_*.json present).
- QuickDraw sketch-rnn: /mnt/1tb/data/quickdraw/sketchrnn/{cat}.{train,valid}.{ptr,strokes}.npy (1035 cats; all Set B present).
- CLIP ViT-B/32: clip_ddetr_*/checkpoints/clip_model/vit_clip2.pth ; text_embeddings.pkl.

## Holdout Set B (14, i%4==0 over 56-cat canonical order — verified == handover list)
backpack, bicycle, clock, couch, dog, elephant, knife, mouse, oven, pizza, sandwich, skateboard, stop sign, train

## Eval / determinism
- seed-14, single-query (n_sketches=1, override of reference default 3), binary/class-agnostic, pycocotools 12-stat.
- determinism: CUBLAS_WORKSPACE_CONFIG=:4096:8, cudnn.deterministic=True, benchmark=False,
  use_deterministic_algorithms(True), seed set BEFORE build_model.

## TO BE FILLED at run time (per OW run)
- wandb_name: <tbd>   output_dir: <tbd>   git/commit of port code: <tbd>
- design-log decisions: see design_log/DESIGN_LOG.md (D1 CDN, D2 LFT, D3 class-head, D4 query-sel, D6 encoder-free).
