#!/bin/bash
set -euo pipefail

RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
DATA_PATH="${DATA_PATH:-FloorSet}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints}"
LOG_DIR="${LOG_DIR:-.}"
WINDOW_START="${WINDOW_START:--1}"
VAL_START="${VAL_START:--1}"
NUM_SAMPLES="${NUM_SAMPLES:-1000000}"
EPOCHS="${EPOCHS:-10}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
LAYERS="${LAYERS:-6}"
DROPOUT="${DROPOUT:-0.10}"
LR="${LR:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-3e-4}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-32}"
VAL_SAMPLES="${VAL_SAMPLES:-20000}"
ENCODER="${ENCODER:-mpnn}"
NUM_HEADS="${NUM_HEADS:-8}"
ORDER_PAIRS="${ORDER_PAIRS:-8192}"
PAIRWISE_PAIRS="${PAIRWISE_PAIRS:-4096}"
PAIRWISE_WEIGHT="${PAIRWISE_WEIGHT:-0.20}"
ANCHOR_WEIGHT="${ANCHOR_WEIGHT:-1.0}"
ASPECT_WEIGHT="${ASPECT_WEIGHT:-0.12}"
PRIORITY_WEIGHT="${PRIORITY_WEIGHT:-0.08}"
ORDER_WEIGHT="${ORDER_WEIGHT:-0.25}"
EDGE_WEIGHT="${EDGE_WEIGHT:-0.08}"
BOUNDARY_WEIGHT_BOOST="${BOUNDARY_WEIGHT_BOOST:-0.45}"
CLUSTER_WEIGHT_BOOST="${CLUSTER_WEIGHT_BOOST:-0.20}"
MIB_WEIGHT_BOOST="${MIB_WEIGHT_BOOST:-0.25}"
CLEAR_RATIO="${CLEAR_RATIO:-1.8}"
MIN_ORDER_GAP="${MIN_ORDER_GAP:-0.025}"
ORDER_TEMP="${ORDER_TEMP:-0.08}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
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
WANDB_PROJECT="${WANDB_PROJECT:-floorset-v11-diffusion}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

DATE_STAMP=$(date +%m%d)
ENCODER_TAG="${ENCODER//-/_}"
DEFAULT_TAG="${DATE_STAMP}_ns${NUM_SAMPLES}_ep${EPOCHS}_enc${ENCODER_TAG}_h${HIDDEN_DIM}_l${LAYERS}_acc${ACCUMULATION_STEPS}"
if [ "$ENCODER" = "graph-transformer" ]; then
  DEFAULT_TAG="${DEFAULT_TAG}_heads${NUM_HEADS}"
fi

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
  --encoder "$ENCODER" \
  --num-heads "$NUM_HEADS" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --grad-clip "$GRAD_CLIP" \
  --accumulation-steps "$ACCUMULATION_STEPS" \
  --anchor-weight "$ANCHOR_WEIGHT" \
  --aspect-weight "$ASPECT_WEIGHT" \
  --priority-weight "$PRIORITY_WEIGHT" \
  --order-weight "$ORDER_WEIGHT" \
  --edge-weight "$EDGE_WEIGHT" \
  --order-pairs "$ORDER_PAIRS" \
  --pairwise-pairs "$PAIRWISE_PAIRS" \
  --pairwise-weight "$PAIRWISE_WEIGHT" \
  --boundary-weight-boost "$BOUNDARY_WEIGHT_BOOST" \
  --cluster-weight-boost "$CLUSTER_WEIGHT_BOOST" \
  --mib-weight-boost "$MIB_WEIGHT_BOOST" \
  --clear-ratio "$CLEAR_RATIO" \
  --min-order-gap "$MIN_ORDER_GAP" \
  --order-temp "$ORDER_TEMP" \
  --clean-sample-policy "$CLEAN_SAMPLE_POLICY" \
  --dirty-sample-weight "$DIRTY_SAMPLE_WEIGHT" \
  --dirty-order-weight "$DIRTY_ORDER_WEIGHT" \
  --num-workers "$NUM_WORKERS" \
  --checkpoint-prefix "$CHECKPOINT_PREFIX" \
  --print-every "$PRINT_EVERY" \
  "${EXTRA_ARGS[@]}" | tee "$LOG_DIR/train_arch_v11_${LOG_TAG}.log"
