#!/bin/bash
# PARTNER_GPU_ARM paired A/B: 3.5 tier (promoted config + RK, steps=10) and
# goal tier (st2 config; TS0=0.05 unlocks the carve gate at 0.44s budgets).
ROOT="/nashome/NVL4/vdalab/yyds-dev/Data-Driven-SoC-Floorplanning"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$ROOT/FloorSet/iccad2026contest:$ROOT/FloorSet"
export DIRECT_CKPT="$ROOT/partner/checkpoints/direct_v2_cont/eval_step1p2M.pt"
export FLOW_CKPT="/nashome/NVL4/vdalab/yyds-dev/codex-worktrees/flow-matching-f1-f3/checkpoints/flow_matching_v1/final.pt"
export PARTNER_FLOW_SLOTS=10 PARTNER_FLOW_STEPS=8 PARTNER_FLOW_SOLVER=euler PARTNER_FLOW_ANTITHETIC=1
export VKILL_OFF=1 PARTNER_PRESCREEN_V=1 PARTNER_OVERSAMPLE=4 PARTNER_TAG_ANCHOR_EXTRA=3
export PARTNER_DIRECT_SOLVER=dpmpp PARTNER_REFINE_STALL_STOP=1
export PARTNER_SA_KERNEL=numba PARTNER_REFINE_KERNEL=numba

run_arm() {
  local name="$1"; shift
  echo "=== $name start $(date '+%H:%M:%S') env: $* ==="
  (
    for kv in "$@"; do export "$kv"; done
    cd "$ROOT/FloorSet/iccad2026contest" && \
    uv run python "$ROOT/scripts/iccad2026_evaluate.py" \
      --data-path ../ --evaluate "$ROOT/partner/contest_optimizer.py" \
      --output "$ROOT/artifacts/partner_eval/${name}.json" 2>&1 | tail -4
  )
  echo "=== $name end $(date '+%H:%M:%S') ==="
}

T35="PARTNER_BUDGET_MAX=3.5 PARTNER_DDIM_STEPS=10 PARTNER_DIRECT_MIN=2.0 PARTNER_NREF=15"
GOAL="PARTNER_POOL_GATE=0 PARTNER_BUDGET_TAU=12 PARTNER_BUDGET_MIN=0.05 PARTNER_BUDGET_MAX=0.75 PARTNER_BUDGET_SCALE=5.0e-5 PARTNER_NREF=6 PARTNER_DIRECT_MIN=0.3 PARTNER_DDIM_STEPS=2"

run_arm ga35_ctrl_a $T35
run_arm ga35_on_a   $T35 PARTNER_GPU_ARM=1
run_arm ga35_ctrl_b $T35
run_arm ga35_on_b   $T35 PARTNER_GPU_ARM=1
run_arm gag_ctrl_a  $GOAL
run_arm gag_on_a    $GOAL PARTNER_GPU_ARM=1 PARTNER_GPU_ARM_TS0=0.05
run_arm gag_ctrl_b  $GOAL
run_arm gag_on_b    $GOAL PARTNER_GPU_ARM=1 PARTNER_GPU_ARM_TS0=0.05
run_arm gag_on_a25  $GOAL PARTNER_GPU_ARM=1 PARTNER_GPU_ARM_TS0=0.05 PARTNER_GPU_ARM_MIN_A=0.25
echo "=== chain done $(date '+%H:%M:%S') ==="
