#!/bin/bash
set -euo pipefail

RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
DATA_PATH="${DATA_PATH:-FloorSet}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints}"
LOG_DIR="${LOG_DIR:-.}"
WINDOW_START="${WINDOW_START:--1}"
VAL_START="${VAL_START:--1}"
NUM_SAMPLES="${NUM_SAMPLES:-800000}"
EPOCHS="${EPOCHS:-4}"
HIDDEN_DIM="${HIDDEN_DIM:-192}"
LAYERS="${LAYERS:-6}"
DROPOUT="${DROPOUT:-0.08}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-2e-4}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-32}"
VAL_SAMPLES="${VAL_SAMPLES:-10000}"
ORDER_PAIRS="${ORDER_PAIRS:-8192}"
PAIRWISE_PAIRS="${PAIRWISE_PAIRS:-4096}"
PAIRWISE_WEIGHT="${PAIRWISE_WEIGHT:-0.20}"
CLEAN_SAMPLE_POLICY="${CLEAN_SAMPLE_POLICY:-weighted}"
DIRTY_SAMPLE_WEIGHT="${DIRTY_SAMPLE_WEIGHT:-0.25}"
DIRTY_ORDER_WEIGHT="${DIRTY_ORDER_WEIGHT:-0.0}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PRINT_EVERY="${PRINT_EVERY:-1000}"
CHECKPOINT_PREFIX="${CHECKPOINT_PREFIX:-gnn}"
WRITE_STABLE_CHECKPOINTS="${WRITE_STABLE_CHECKPOINTS:-0}"
IGNORE_OPTIMIZER_STATE="${IGNORE_OPTIMIZER_STATE:-0}"
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-floorset-arch-v4}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

DATE_STAMP=$(date +%m%d)
DEFAULT_TAG="${DATE_STAMP}_ns${NUM_SAMPLES}_ep${EPOCHS}_h${HIDDEN_DIM}_l${LAYERS}_acc${ACCUMULATION_STEPS}"

CHECKPOINT_TAG="${CHECKPOINT_TAG:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-$DEFAULT_TAG}"

LOG_TAG="$DEFAULT_TAG"

EXTRA_ARGS=()
if [ "$WANDB" = "1" ]; then
  EXTRA_ARGS+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-mode "$WANDB_MODE")
  if [ -n "$WANDB_ENTITY" ]; then
    EXTRA_ARGS+=(--wandb-entity "$WANDB_ENTITY")
  fi
  if [ -n "$WANDB_RUN_NAME" ]; then
    EXTRA_ARGS+=(--wandb-run-name "$WANDB_RUN_NAME")
  fi
fi
if [ -n "$CHECKPOINT_TAG" ]; then
  EXTRA_ARGS+=(--checkpoint-tag "$CHECKPOINT_TAG")
fi
if [ -n "$RESUME_CHECKPOINT" ]; then
  EXTRA_ARGS+=(--resume-checkpoint "$RESUME_CHECKPOINT")
fi
if [ "$WRITE_STABLE_CHECKPOINTS" = "1" ]; then
  EXTRA_ARGS+=(--write-stable-checkpoints)
fi
if [ "$IGNORE_OPTIMIZER_STATE" = "1" ]; then
  EXTRA_ARGS+=(--ignore-optimizer-state)
fi

mkdir -p "$LOG_DIR"

uv run -m floorset_arch.training.train \
  --data-path "$DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --window-start "$WINDOW_START" \
  --val-start "$VAL_START" \
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
  --clean-sample-policy "$CLEAN_SAMPLE_POLICY" \
  --dirty-sample-weight "$DIRTY_SAMPLE_WEIGHT" \
  --dirty-order-weight "$DIRTY_ORDER_WEIGHT" \
  --num-workers "$NUM_WORKERS" \
  --checkpoint-prefix "$CHECKPOINT_PREFIX" \
  --print-every "$PRINT_EVERY" \
  "${EXTRA_ARGS[@]}" | tee "$LOG_DIR/train_arch_v4_${LOG_TAG}.log"
