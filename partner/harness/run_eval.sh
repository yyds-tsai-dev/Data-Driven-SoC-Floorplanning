#!/bin/bash
# Evaluate the partner column-slicing solver (v2, heuristic seed only) with
# the same local evaluator + data setup the repo eval scripts use.
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PATH="/nashome/NVL4/vdalab/yyds-dev/.local/bin:$PATH"
export PYTHONPATH="$ROOT/FloorSet/iccad2026contest:$ROOT/FloorSet:${PYTHONPATH:-}"
cd "$ROOT/FloorSet/iccad2026contest"
exec uv run "$ROOT/scripts/iccad2026_evaluate.py" \
  --data-path ../ \
  --evaluate "$ROOT/partner/harness/my_opt_claude_v2.py" \
  --verbose \
  "$@"
