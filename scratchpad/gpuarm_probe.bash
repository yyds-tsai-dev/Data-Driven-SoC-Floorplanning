#!/bin/bash
ROOT="/nashome/NVL4/vdalab/yyds-dev/Data-Driven-SoC-Floorplanning"
WT="$ROOT/.claude/worktrees/agent-a4384881f2b3c4db6"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$ROOT/FloorSet/iccad2026contest:$ROOT/FloorSet"
export DIRECT_CKPT="$ROOT/partner/checkpoints/direct_v2_cont/eval_step1p2M.pt"
export VKILL_OFF=1 PARTNER_PRESCREEN_V=1 PARTNER_NREF=15
export PARTNER_OVERSAMPLE=4 PARTNER_TAG_ANCHOR_EXTRA=3
export PARTNER_DIRECT_SOLVER=dpmpp PARTNER_DDIM_STEPS=10
export PARTNER_REFINE_STALL_STOP=1
export PARTNER_REFINE_KERNEL=numba PARTNER_REFINE_FASTBUILD=1
export PARTNER_POOL=8
export PARTNER_BUDGET_MAX="${2:-3.5}" PARTNER_DIRECT_MIN=2.0 PARTNER_POOL_GATE=3.0
export PARTNER_GPU_ARM_DEBUG=1
if [ "${3:-on}" = "on" ]; then export PARTNER_GPU_ARM=1; fi
TID="${1:-95}"
cd "$ROOT/FloorSet/iccad2026contest"
uv run --project "$WT" python "$ROOT/scripts/iccad2026_evaluate.py" \
  --data-path ../ --evaluate "$WT/partner/contest_optimizer.py" \
  --test-id "$TID" 2>&1
