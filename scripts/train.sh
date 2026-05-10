#!/bin/bash
set -euo pipefail

RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
WINDOW_START="${WINDOW_START:--1}"
NUM_SAMPLES="${NUM_SAMPLES:-500000}"
EPOCHS="${EPOCHS:-4}"
CHECKPOINT_TAG="${CHECKPOINT_TAG:-}"
HIDDEN_DIM="${HIDDEN_DIM:-192}"
LAYERS="${LAYERS:-6}"
DROPOUT="${DROPOUT:-0.08}"
LR="${LR:-4e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-2e-4}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-32}"
VAL_SAMPLES="${VAL_SAMPLES:-5000}"
ORDER_PAIRS="${ORDER_PAIRS:-8192}"
LOG_TAG="${CHECKPOINT_TAG:-full1m_$(date +%m%d)}"

EXTRA_ARGS=()
if [ -n "$CHECKPOINT_TAG" ]; then
  EXTRA_ARGS+=(--checkpoint-tag "$CHECKPOINT_TAG")
fi
if [ -n "$RESUME_CHECKPOINT" ]; then
  EXTRA_ARGS+=(--resume-checkpoint "$RESUME_CHECKPOINT")
fi

uv run -m floorset_arch.training.train \
  --data-path FloorSet \
  --output-dir checkpoints \
  --window-start "$WINDOW_START" \
  --num-samples "$NUM_SAMPLES" \
  --val-samples "$VAL_SAMPLES" \
  --epochs "$EPOCHS" \
  --device cuda \
  --hidden-dim "$HIDDEN_DIM" \
  --layers "$LAYERS" \
  --dropout "$DROPOUT" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --accumulation-steps "$ACCUMULATION_STEPS" \
  --order-pairs "$ORDER_PAIRS" \
  --wandb \
  --wandb-project floorset-arch-v2 \
  --print-every 500 \
  "${EXTRA_ARGS[@]}" | tee "train_arch_v3_${LOG_TAG}.log"
