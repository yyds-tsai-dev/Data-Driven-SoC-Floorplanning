#!/bin/bash
TESTID="${1:-95}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export FLOORSET_GNN_CHECKPOINT="${FLOORSET_GNN_CHECKPOINT:-$ROOT/checkpoints/gnn_best.pt}"
cd FloorSet/iccad2026contest
uv run iccad2026_evaluate.py \
  --evaluate "$ROOT/src/architecture_v2_optimizer.py" \
  --test-id $TESTID \
  --verbose
