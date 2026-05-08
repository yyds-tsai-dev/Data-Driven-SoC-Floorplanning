#!/bin/bash

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ -z "${FLOORSET_V1_CHECKPOINT:-}" ] && [ -f "$ROOT/checkpoints/nvl4_v1.pt" ]; then
  export FLOORSET_V1_CHECKPOINT="$ROOT/checkpoints/nvl4_v1.pt"
fi
cd "$ROOT/FloorSet/iccad2026contest"
uv run iccad2026_evaluate.py \
  --evaluate "$ROOT/src/architecture_v1_optimizer.py"
