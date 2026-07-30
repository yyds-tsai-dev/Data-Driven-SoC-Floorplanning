#!/bin/bash
# Budget tail scan: PARTNER_BUDGET_MAX in {8,6,4}, all else = scoring env.
# Baseline (MAX=24) = artifacts/partner_eval/cont_retrieval_direct_control.json
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$ROOT/FloorSet/iccad2026contest:$ROOT/FloorSet"
export DIRECT_CKPT="$ROOT/partner/checkpoints/direct_v2_cont/eval_retrieval_direct_control.pt"
export VKILL_OFF=1 PARTNER_PRESCREEN_V=1 PARTNER_NREF=15 \
       PARTNER_OVERSAMPLE=4 PARTNER_TAG_ANCHOR_EXTRA=3
cd "$ROOT/FloorSet/iccad2026contest"
for MAX in 8 6 4; do
  echo "=== PARTNER_BUDGET_MAX=$MAX ==="
  PARTNER_BUDGET_MAX=$MAX uv run python "$ROOT/scripts/iccad2026_evaluate.py" \
    --data-path ../ \
    --evaluate "$ROOT/partner/contest_optimizer.py" \
    --output "$ROOT/artifacts/partner_eval/budget_scan_max${MAX}.json" \
    2>&1 | tail -3
done
echo "=== budget scan complete ==="
