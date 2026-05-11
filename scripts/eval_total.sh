#!/bin/bash

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Usage:
#   ./script.sh                 # default: gnn_best.pt
#   ./script.sh gnn_epoch10.pt
#   ./script.sh /path/to/model.pt

CKPT_NAME="${1:-gnn_best.pt}"

if [[ "$CKPT_NAME" = /* ]]; then
  CKPT_PATH="$CKPT_NAME"
else
  CKPT_PATH="$ROOT/checkpoints/$CKPT_NAME"
fi

if [ -z "${FLOORSET_GNN_CHECKPOINT:-}" ] && [ -f "$CKPT_PATH" ]; then
  export FLOORSET_GNN_CHECKPOINT="$CKPT_PATH"
fi

cd "$ROOT/FloorSet/iccad2026contest"

uv run iccad2026_evaluate.py \
  --evaluate "$ROOT/src/architecture_v3_optimizer.py" \
  --verbose