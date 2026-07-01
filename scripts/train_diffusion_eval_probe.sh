#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
DATA_PATH="${DATA_PATH:-FloorSet}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints}"
LOG_DIR="${LOG_DIR:-.}"
DATASET_MODE="${DATASET_MODE:-eval-probe}"
WINDOW_START="${WINDOW_START:--1}"
VAL_START="${VAL_START:--1}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
VAL_SAMPLES="${VAL_SAMPLES:-100}"
EPOCHS="${EPOCHS:-100}"
VARIANT="${VARIANT:-hgt_lite}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
LAYERS="${LAYERS:-2}"
LR="${LR:-1.5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-3e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-32}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-1000}"
NOISE_SCHEDULE="${NOISE_SCHEDULE:-cosine}"
BETA_START="${BETA_START:-1e-4}"
BETA_END="${BETA_END:-0.02}"
NOISE_SAMPLES="${NOISE_SAMPLES:-1}"
EMA_DECAY="${EMA_DECAY:-0.9999}"
PAIR_WEIGHT="${PAIR_WEIGHT:-0.25}"
TREE_WEIGHT="${TREE_WEIGHT:-0}"
QUALITY_WEIGHT="${QUALITY_WEIGHT:-0.01}"
ASPECT_WEIGHT="${ASPECT_WEIGHT:-0.05}"
OVERLAP_WEIGHT="${OVERLAP_WEIGHT:-0.05}"
BBOX_WEIGHT="${BBOX_WEIGHT:-0.01}"
NET_WEIGHT="${NET_WEIGHT:-0.01}"
CLUSTER_WEIGHT="${CLUSTER_WEIGHT:-0.02}"
BOUNDARY_WEIGHT="${BOUNDARY_WEIGHT:-0.02}"
MIB_WEIGHT="${MIB_WEIGHT:-0.02}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PRINT_EVERY="${PRINT_EVERY:-1000}"
CHECKPOINT_PREFIX="${CHECKPOINT_PREFIX:-diffusion_eval_probe}"
TRAIN_EVALUATE_EACH_EPOCH="${TRAIN_EVALUATE_EACH_EPOCH:-0}"
SYNTHETIC_SMOKE_SAMPLES="${SYNTHETIC_SMOKE_SAMPLES:-0}"
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-floorset-v11-diffusion}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

DATE_STAMP=$(date +%m%d)
VARIANT_TAG="${VARIANT//-/_}"
DEFAULT_TAG="${DATE_STAMP}_probe_ns${NUM_SAMPLES}_ep${EPOCHS}_diff${VARIANT_TAG}_h${HIDDEN_DIM}_l${LAYERS}_steps${DIFFUSION_STEPS}_acc${ACCUMULATION_STEPS}_bs${BATCH_SIZE}"

CHECKPOINT_TAG="${CHECKPOINT_TAG:-$DEFAULT_TAG}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-$CHECKPOINT_TAG}"
LOG_TAG="${LOG_TAG:-$CHECKPOINT_TAG}"
EVAL_OUTPUT="${EVAL_OUTPUT:-artifacts/eval_probe/${LOG_TAG}_full_eval.json}"
PROBE_CKPT="$OUTPUT_DIR/${CHECKPOINT_PREFIX}_latest_${CHECKPOINT_TAG}.pt"

if [ "$DATASET_MODE" != "eval-probe" ]; then
  echo "DATASET_MODE must remain eval-probe for this script; got: $DATASET_MODE" >&2
  exit 2
fi

case "$TREE_WEIGHT" in
  0|0.0|0.00|0.000|0e0|0E0) ;;
  *)
    echo "TREE_WEIGHT must remain zero for eval-probe training; got: $TREE_WEIGHT" >&2
    exit 2
    ;;
esac

if [ "$TRAIN_EVALUATE_EACH_EPOCH" != "0" ]; then
  echo "TRAIN_EVALUATE_EACH_EPOCH must stay 0; this script runs one final evaluator pass." >&2
  exit 2
fi

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
if [ -n "$RESUME_CHECKPOINT" ]; then
  EXTRA_ARGS+=(--resume-checkpoint "$RESUME_CHECKPOINT")
fi
if [ "$SYNTHETIC_SMOKE_SAMPLES" != "0" ]; then
  EXTRA_ARGS+=(--synthetic-smoke-samples "$SYNTHETIC_SMOKE_SAMPLES")
fi

mkdir -p "$LOG_DIR"
mkdir -p "$(dirname "$EVAL_OUTPUT")"

uv run -m floorset_arch.training.train_diffusion \
  --dataset-mode "$DATASET_MODE" \
  --data-path "$DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --window-start "$WINDOW_START" \
  --val-start "$VAL_START" \
  --num-samples "$NUM_SAMPLES" \
  --val-samples "$VAL_SAMPLES" \
  --epochs "$EPOCHS" \
  --device "$DEVICE" \
  --variant "$VARIANT" \
  --hidden-dim "$HIDDEN_DIM" \
  --layers "$LAYERS" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --grad-clip "$GRAD_CLIP" \
  --accumulation-steps "$ACCUMULATION_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --max-diffusion-steps "$DIFFUSION_STEPS" \
  --noise-schedule "$NOISE_SCHEDULE" \
  --beta-start "$BETA_START" \
  --beta-end "$BETA_END" \
  --noise-samples "$NOISE_SAMPLES" \
  --ema-decay "$EMA_DECAY" \
  --pair-weight "$PAIR_WEIGHT" \
  --tree-weight "$TREE_WEIGHT" \
  --quality-weight "$QUALITY_WEIGHT" \
  --aspect-weight "$ASPECT_WEIGHT" \
  --overlap-weight "$OVERLAP_WEIGHT" \
  --bbox-weight "$BBOX_WEIGHT" \
  --net-weight "$NET_WEIGHT" \
  --cluster-weight "$CLUSTER_WEIGHT" \
  --boundary-weight "$BOUNDARY_WEIGHT" \
  --mib-weight "$MIB_WEIGHT" \
  --num-workers "$NUM_WORKERS" \
  --checkpoint-prefix "$CHECKPOINT_PREFIX" \
  --checkpoint-tag "$CHECKPOINT_TAG" \
  --print-every "$PRINT_EVERY" \
  "${EXTRA_ARGS[@]}" | tee "$LOG_DIR/train_arch_v11_diffusion_eval_probe_${LOG_TAG}.log"

bash "$ROOT/scripts/eval_total.sh" --diffusion-checkpoint "$PROBE_CKPT" --diffusion-use-ema --output "$EVAL_OUTPUT"
