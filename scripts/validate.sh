#!/bin/bash

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/FloorSet/iccad2026contest"
uv run iccad2026_evaluate.py \
  --validate "$ROOT/src/architecture_v5_optimizer.py"
