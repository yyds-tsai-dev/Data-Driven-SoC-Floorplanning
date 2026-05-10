#!/bin/bash
set -euo pipefail

RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-checkpoints/gnn_latest.pt}"
WINDOW_START="${WINDOW_START:--1}"
NUM_SAMPLES="${NUM_SAMPLES:-500000}"
EPOCHS="${EPOCHS:-1}"
CHECKPOINT_TAG="${CHECKPOINT_TAG:-}"
LOG_TAG="${CHECKPOINT_TAG:-resume_900k_$(date +%m%d)}"

EXTRA_ARGS=()
if [ -n "$CHECKPOINT_TAG" ]; then
  EXTRA_ARGS+=(--checkpoint-tag "$CHECKPOINT_TAG")
fi

uv run -m floorset_arch.training.train \
  --data-path FloorSet \
  --output-dir checkpoints \
  --resume-checkpoint "$RESUME_CHECKPOINT" \
  --window-start "$WINDOW_START" \
  --num-samples "$NUM_SAMPLES" \
  --val-samples 2000 \
  --epochs "$EPOCHS" \
  --device cuda \
  --hidden-dim 160 \
  --layers 5 \
  --lr 6e-4 \
  --accumulation-steps 16 \
  --wandb \
  --wandb-project floorset-arch-v2 \
  --print-every 500 \
  "${EXTRA_ARGS[@]}" | tee "train_arch_v2_${LOG_TAG}.log"
