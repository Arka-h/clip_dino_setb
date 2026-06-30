#!/bin/bash
# ===========================================================================
# HEADLINE RUN (#9) — DINO-CASF, OW (Set B held out), cos34, 50 epochs, 100% data.
# Effective batch 16 (canonical-match, per user). Local logs only (no W&B).
# Gate 2 PASSED before this is meant to run (design_log/GATE2_SANITY_LOG.md).
#
# NOT auto-run. Launch when ready:  bash scripts/run_full_cos34_ow.sh
# Resumable: re-running resumes from outputs/casf_ow_cos34_b16/checkpoint.pth if --resume added.
# Expect ~4-5 days on one Quadro RTX 8000.
# ===========================================================================
set -e
cd "$(dirname "$0")/.."
source /home/rahul/miniconda3/etc/profile.d/conda.sh
conda activate clip_ddetr
export CUDA_HOME=/usr/local/cuda-12.8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0

OUT=outputs/casf_ow_cos34_b16
mkdir -p "$OUT" logs

# micro-batch 4 + accum 4 = effective 16 (safe on a shared GPU; bump --microbatch to 8
# and the entrypoint auto-sets accum=2 if you have the GPU to yourself — probe: bs8 ~24GB).
nohup python -u casf/train_casf.py \
  --sched cos34 \
  --data_frac 1.0 \
  --epochs 50 \
  --microbatch 4 --effective_batch 16 \
  --output_dir "$OUT" \
  --wandb_name dino_casf_ow_cos34_b16 \
  --seed 42 \
  --num_workers 4 \
  > logs/full_cos34_ow.log 2>&1 &

echo "launched full cos34 OW run, PID $!  ->  tail -f logs/full_cos34_ow.log"
echo "output dir: $OUT   (checkpoint.pth saved each epoch; provenance.json written once)"
# Notes:
#  - Full seed-14 OW+CW 12-stat eval runs at the END (no --max_eval_imgs cap => all val imgs).
#    For periodic eval, run `python casf/train_casf.py --eval_only --resume $OUT/checkpoint.pth`
#    in a separate process against the latest checkpoint.
#  - Train-time determinism: use_deterministic_algorithms is warn_only=True (matches the reference,
#    which gated strict determinism on --eval). EVAL determinism is bit-identical (Gate 1). A
#    MemoryEfficientAttention non-deterministic *backward* warning is expected and does not affect
#    the eval AP; forcing warn_only=False risks erroring on deformable-attn backward.
