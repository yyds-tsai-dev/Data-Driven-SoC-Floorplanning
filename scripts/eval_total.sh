#!/bin/bash

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ -z "${FLOORSET_GNN_CHECKPOINT:-}" ] && [ -f "$ROOT/checkpoints/gnn_best.pt" ]; then
  export FLOORSET_GNN_CHECKPOINT="$ROOT/checkpoints/gnn_best.pt"
fi
cd "$ROOT/FloorSet/iccad2026contest"
uv run iccad2026_evaluate.py \
  --evaluate "$ROOT/src/architecture_v2_optimizer.py" \
  --verbose
