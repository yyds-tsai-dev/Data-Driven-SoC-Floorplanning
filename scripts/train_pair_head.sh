#!/bin/bash
# Pair-head v2 retraining: warm-start the encoder from a v1 anchor checkpoint,
# train a fresh 4-class order-faithful pair head (phase 1 frozen encoder, phase 2
# low-LR joint refine). The metric that matters is held-out composite
# axis+direction agreement vs the decoder's build_order_dags rule, printed as
# pair_acc (composite) and pair_axis_acc (axis-only) per epoch.
#
# UPPER_CASE env overrides mirror scripts/train.sh; PAIR_* toggles are new.
set -euo pipefail

# Warm-start encoder from the 0514 v1 checkpoint (strict=False; fresh v2 head).
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-checkpoints/gnn_best_0514_ns800000_ep4_h192_l6_acc32.pt}"
DATA_PATH="${DATA_PATH:-FloorSet}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints}"
LOG_DIR="${LOG_DIR:-.}"
WINDOW_START="${WINDOW_START:--1}"
VAL_START="${VAL_START:--1}"
# Fast-iteration model: a 200k window x a few epochs, not a marathon.
NUM_SAMPLES="${NUM_SAMPLES:-200000}"
EPOCHS="${EPOCHS:-4}"
VAL_SAMPLES="${VAL_SAMPLES:-4000}"
# Encoder geometry MUST match the resume checkpoint (0514 = h192/l6/mpnn); the
# resume loader overrides these from the payload anyway, but keep them honest.
HIDDEN_DIM="${HIDDEN_DIM:-192}"
LAYERS="${LAYERS:-6}"
DROPOUT="${DROPOUT:-0.05}"
ENCODER="${ENCODER:-mpnn}"
NUM_HEADS="${NUM_HEADS:-8}"
# Joint-phase base LR (used only if two-phase is off); phase LRs below dominate.
LR="${LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-32}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"

# Anchor/priority/aspect/order/edge losses kept at their 0514 weights so the
# OTHER hint channel (anchors) does not regress during the joint phase.
ANCHOR_WEIGHT="${ANCHOR_WEIGHT:-1.0}"
ASPECT_WEIGHT="${ASPECT_WEIGHT:-0.12}"
PRIORITY_WEIGHT="${PRIORITY_WEIGHT:-0.08}"
ORDER_WEIGHT="${ORDER_WEIGHT:-0.25}"
EDGE_WEIGHT="${EDGE_WEIGHT:-0.08}"
BOUNDARY_WEIGHT_BOOST="${BOUNDARY_WEIGHT_BOOST:-0.45}"
CLUSTER_WEIGHT_BOOST="${CLUSTER_WEIGHT_BOOST:-0.20}"
MIB_WEIGHT_BOOST="${MIB_WEIGHT_BOOST:-0.25}"
ORDER_PAIRS="${ORDER_PAIRS:-8192}"

# --- Pair-head v2 tunables ---
# Large pair batches: all i<j pairs for typical n<=120 fit comfortably.
PAIRWISE_PAIRS="${PAIRWISE_PAIRS:-8192}"
PAIR_WEIGHT="${PAIR_WEIGHT:-1.0}"        # CE weight on the 4-class head
PAIR_EDGE_ALPHA="${PAIR_EDGE_ALPHA:-0.25}"  # per-pair 1 + alpha*net_weight
PAIR_TIE_ALPHA="${PAIR_TIE_ALPHA:-0.5}"     # up-weight hard near-ties
PAIR_CLASS_WEIGHT="${PAIR_CLASS_WEIGHT:-}"  # optional "wx0,wx1,wy0,wy1"
PAIR_PHASE1_EPOCHS="${PAIR_PHASE1_EPOCHS:-2}"  # frozen-encoder head-only epochs
PAIR_PHASE1_LR="${PAIR_PHASE1_LR:-1e-3}"
PAIR_PHASE2_LR="${PAIR_PHASE2_LR:-2e-5}"

CLEAN_SAMPLE_POLICY="${CLEAN_SAMPLE_POLICY:-weighted}"
DIRTY_SAMPLE_WEIGHT="${DIRTY_SAMPLE_WEIGHT:-0.25}"
DIRTY_ORDER_WEIGHT="${DIRTY_ORDER_WEIGHT:-0.0}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PRINT_EVERY="${PRINT_EVERY:-1000}"
CHECKPOINT_PREFIX="${CHECKPOINT_PREFIX:-gnn_pairv2}"
IGNORE_OPTIMIZER_STATE="${IGNORE_OPTIMIZER_STATE:-1}"  # v2 head changes param set
WANDB="${WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-floorset-pair-head-v2}"
WANDB_MODE="${WANDB_MODE:-online}"

DATE_STAMP=$(date +%m%d)
ENCODER_TAG="${ENCODER//-/_}"
DEFAULT_TAG="${DATE_STAMP}_pairv2_ns${NUM_SAMPLES}_ep${EPOCHS}_enc${ENCODER_TAG}_h${HIDDEN_DIM}_l${LAYERS}_acc${ACCUMULATION_STEPS}"
CHECKPOINT_TAG="${CHECKPOINT_TAG:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-$DEFAULT_TAG}"
LOG_TAG="$DEFAULT_TAG"

EXTRA_ARGS=(--pair-v2)
if [ "$WANDB" = "1" ]; then
  EXTRA_ARGS+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-mode "$WANDB_MODE")
  if [ -n "$WANDB_RUN_NAME" ]; then EXTRA_ARGS+=(--wandb-run-name "$WANDB_RUN_NAME"); fi
fi
if [ -n "$CHECKPOINT_TAG" ]; then EXTRA_ARGS+=(--checkpoint-tag "$CHECKPOINT_TAG"); fi
if [ -n "$RESUME_CHECKPOINT" ]; then EXTRA_ARGS+=(--resume-checkpoint "$RESUME_CHECKPOINT"); fi
if [ "$IGNORE_OPTIMIZER_STATE" = "1" ]; then EXTRA_ARGS+=(--ignore-optimizer-state); fi
if [ -n "$PAIR_CLASS_WEIGHT" ]; then EXTRA_ARGS+=(--pair-class-weight "$PAIR_CLASS_WEIGHT"); fi

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
  --pair-weight "$PAIR_WEIGHT" \
  --pair-edge-alpha "$PAIR_EDGE_ALPHA" \
  --pair-tie-alpha "$PAIR_TIE_ALPHA" \
  --pair-phase1-epochs "$PAIR_PHASE1_EPOCHS" \
  --pair-phase1-lr "$PAIR_PHASE1_LR" \
  --pair-phase2-lr "$PAIR_PHASE2_LR" \
  --boundary-weight-boost "$BOUNDARY_WEIGHT_BOOST" \
  --cluster-weight-boost "$CLUSTER_WEIGHT_BOOST" \
  --mib-weight-boost "$MIB_WEIGHT_BOOST" \
  --clean-sample-policy "$CLEAN_SAMPLE_POLICY" \
  --dirty-sample-weight "$DIRTY_SAMPLE_WEIGHT" \
  --dirty-order-weight "$DIRTY_ORDER_WEIGHT" \
  --num-workers "$NUM_WORKERS" \
  --checkpoint-prefix "$CHECKPOINT_PREFIX" \
  --print-every "$PRINT_EVERY" \
  "${EXTRA_ARGS[@]}" | tee "$LOG_DIR/train_arch_v11_${LOG_TAG}.log"
