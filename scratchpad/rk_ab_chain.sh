#!/bin/bash
# PARTNER_REFINE_KERNEL decisive A/B (rung /12-23, commit ca9adea + warm 015fe0f):
#   rk_ctrl35 / rk_on35 (x2 pairs) : promoted tier -- n>=110 slots should start
#                                    delivering (they are 15/15 dead today)
#   rk_goal_ctrl / rk_goal_on      : goal point gp020 -- RK also speeds
#                                    refine_positions in column workers
#   rk_goal_dm03                   : direct resurrection with fast rungs at
#                                    0.44s tail budgets (12 rungs now fit)
ROOT="/nashome/NVL4/vdalab/yyds-dev/Data-Driven-SoC-Floorplanning"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$ROOT/FloorSet/iccad2026contest:$ROOT/FloorSet"

export DIRECT_CKPT="$ROOT/partner/checkpoints/direct_v2_cont/eval_step1p2M.pt"
export FLOW_CKPT="/nashome/NVL4/vdalab/yyds-dev/codex-worktrees/flow-matching-f1-f3/checkpoints/flow_matching_v1/final.pt"
export PARTNER_FLOW_SLOTS=10 PARTNER_FLOW_STEPS=8 PARTNER_FLOW_SOLVER=euler PARTNER_FLOW_ANTITHETIC=1
export VKILL_OFF=1 PARTNER_PRESCREEN_V=1 PARTNER_NREF=15 \
       PARTNER_OVERSAMPLE=4 PARTNER_TAG_ANCHOR_EXTRA=3
export PARTNER_DIRECT_SOLVER=dpmpp PARTNER_DDIM_STEPS=10 PARTNER_REFINE_STALL_STOP=1
export PARTNER_DIRECT_MIN=2.0

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

GOAL="PARTNER_SA_KERNEL=numba PARTNER_POOL_GATE=0 PARTNER_BUDGET_SCALE=5.0e-5 PARTNER_BUDGET_TAU=12 PARTNER_BUDGET_MIN=0.05 PARTNER_BUDGET_MAX=0.75"

run_arm rk_ctrl35    PARTNER_BUDGET_MAX=3.5
run_arm rk_on35      PARTNER_BUDGET_MAX=3.5 PARTNER_REFINE_KERNEL=numba
run_arm rk_ctrl35_r2 PARTNER_BUDGET_MAX=3.5
run_arm rk_on35_r2   PARTNER_BUDGET_MAX=3.5 PARTNER_REFINE_KERNEL=numba
run_arm rk_goal_ctrl $GOAL
run_arm rk_goal_on   $GOAL PARTNER_REFINE_KERNEL=numba
run_arm rk_goal_dm03 $GOAL PARTNER_REFINE_KERNEL=numba PARTNER_DIRECT_MIN=0.3 PARTNER_NREF=6
echo "=== chain done $(date '+%H:%M:%S') ==="
