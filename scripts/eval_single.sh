#!/bin/bash
TESTID="${1:-95}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export FLOORSET_V1_CHECKPOINT="${FLOORSET_V1_CHECKPOINT:-$ROOT/checkpoints/nvl4_v1.pt}"
cd FloorSet/iccad2026contest
uv run iccad2026_evaluate.py \
  --evaluate "$ROOT/src/architecture_v1_optimizer.py" \
  --test-id $TESTID \
  --verbose
