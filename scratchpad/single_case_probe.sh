#!/bin/bash
# ONE validation case through the real partner optimizer (pool on), REFINER_DEBUG
ROOT="/nashome/NVL4/vdalab/yyds-dev/Data-Driven-SoC-Floorplanning"
WT="$ROOT/.claude/worktrees/agent-ae70c831b38cdd03d"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$ROOT/FloorSet/iccad2026contest:$ROOT/FloorSet"
export DIRECT_CKPT="$ROOT/partner/checkpoints/direct_v2_cont/eval_step1p2M.pt"
export VKILL_OFF=1 PARTNER_PRESCREEN_V=1 PARTNER_NREF=15 \
       PARTNER_OVERSAMPLE=4 PARTNER_TAG_ANCHOR_EXTRA=3
export PARTNER_DIRECT_SOLVER=dpmpp PARTNER_DDIM_STEPS=10
export PARTNER_REFINE_STALL_STOP=1
export REFINER_DEBUG=1
export PARTNER_SA_WORKERS=16 PARTNER_SA_CONFIGS=16
TID="${1:-95}"; MAXB="${2:-3.5}"; GATE="${3:-3.0}"
export PARTNER_BUDGET_MAX="$MAXB" PARTNER_DIRECT_MIN=2.0 PARTNER_POOL_GATE="$GATE"
cd "$ROOT/FloorSet/iccad2026contest"
uv run python "$ROOT/scripts/iccad2026_evaluate.py" \
  --data-path ../ --evaluate "$WT/partner/contest_optimizer.py" \
  --test-id "$TID" 2>&1
