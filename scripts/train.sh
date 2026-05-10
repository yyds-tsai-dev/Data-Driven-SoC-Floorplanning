#!/bin/bash
set -euo pipefail

RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
WINDOW_START="${WINDOW_START:--1}"
NUM_SAMPLES="${NUM_SAMPLES:-200000}"
EPOCHS="${EPOCHS:-10}"
HIDDEN_DIM="${HIDDEN_DIM:-192}"
LAYERS="${LAYERS:-6}"
DROPOUT="${DROPOUT:-0.08}"
LR="${LR:-4e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-2e-4}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-32}"
VAL_SAMPLES="${VAL_SAMPLES:-5000}"
ORDER_PAIRS="${ORDER_PAIRS:-8192}"
PAIRWISE_PAIRS="${PAIRWISE_PAIRS:-4096}"
PAIRWISE_WEIGHT="${PAIRWISE_WEIGHT:-0.20}"
DEVICE="${DEVICE:-cuda}"

DATE_STAMP=$(date +%m%d)
DEFAULT_TAG="${DATE_STAMP}_ns${NUM_SAMPLES}_ep${EPOCHS}_h${HIDDEN_DIM}_l${LAYERS}_acc${ACCUMULATION_STEPS}"

CHECKPOINT_TAG="${CHECKPOINT_TAG:-}"

LOG_TAG="$DEFAULT_TAG"

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
  --device "$DEVICE" \
  --hidden-dim "$HIDDEN_DIM" \
  --layers "$LAYERS" \
  --dropout "$DROPOUT" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --accumulation-steps "$ACCUMULATION_STEPS" \
  --order-pairs "$ORDER_PAIRS" \
  --pairwise-pairs "$PAIRWISE_PAIRS" \
  --pairwise-weight "$PAIRWISE_WEIGHT" \
  --wandb \
  --wandb-project floorset-arch-v3 \
  --wandb-run-name "$DEFAULT_TAG" \
  --print-every 500 \
  "${EXTRA_ARGS[@]}" | tee "train_arch_v3_${LOG_TAG}.log"
