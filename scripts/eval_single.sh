#!/bin/bash
TESTID="${1:-95}"

export FLOORSET_V1_CHECKPOINT=checkpoints/nvl4_v1.pt
cd FloorSet/iccad2026contest
uv run iccad2026_evaluate.py \
  --evaluate ../../src/architecture_v1_optimizer.py \
  --test-id $TESTID \
  --verbose
