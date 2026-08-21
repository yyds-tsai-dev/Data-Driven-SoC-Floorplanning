#!/bin/bash
set -e
ROOT="/nashome/NVL4/vdalab/yyds-dev/Data-Driven-SoC-Floorplanning"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$ROOT/FloorSet/iccad2026contest:$ROOT/FloorSet"

export DIRECT_CKPT="$ROOT/partner/checkpoints/direct_v2_cont/eval_step1p2M.pt"
export FLOW_CKPT="/nashome/NVL4/vdalab/yyds-dev/codex-worktrees/flow-matching-f1-f3/checkpoints/flow_matching_v1/final.pt"
export PARTNER_FLOW_SLOTS=10 PARTNER_FLOW_STEPS=2 PARTNER_FLOW_SOLVER=euler PARTNER_FLOW_ANTITHETIC=1
export VKILL_OFF=1 PARTNER_PRESCREEN_V=1 PARTNER_NREF=15 \
       PARTNER_OVERSAMPLE=4 PARTNER_TAG_ANCHOR_EXTRA=3
export PARTNER_BUDGET_MAX=3.5 PARTNER_DDIM_STEPS=25 PARTNER_DIRECT_MIN=2.0 PARTNER_DIRECT_SOLVER=dpmpp PARTNER_DDIM_STEPS=10

OUT="$ROOT/artifacts/partner_eval/dpmpp10_flowst2.json"
cd "$ROOT/FloorSet/iccad2026contest"
echo "FLOWST2 -> $OUT"
uv run python "$ROOT/scripts/iccad2026_evaluate.py" \
  --data-path ../ --evaluate "$ROOT/partner/my_opt_claude.py" \
  --output "$OUT" 2>&1 | tail -20
