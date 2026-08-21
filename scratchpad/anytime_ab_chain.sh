#!/bin/bash
# PARTNER_ANYTIME_LADDER paired A/B (commit bceac97):
#   al_ctrl20/al_on20 : judgment tier MAX=2.0 gate0 (gate <= -0.01 vs control)
#   al_ctrl35/al_on35 : promoted tier (must not regress; n>=110 slot waste fix)
#   al_goal_dm03      : goal tier + DM=0.3 + NREF=6 -- can the anytime ladder
#                       resurrect the direct channel at 0.55s tail budgets?
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
export PARTNER_SA_KERNEL=numba

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

run_arm al_ctrl20    PARTNER_BUDGET_MAX=2.0 PARTNER_POOL_GATE=0
run_arm al_on20      PARTNER_BUDGET_MAX=2.0 PARTNER_POOL_GATE=0 PARTNER_ANYTIME_LADDER=1
run_arm al_ctrl35    PARTNER_BUDGET_MAX=3.5
run_arm al_on35      PARTNER_BUDGET_MAX=3.5 PARTNER_ANYTIME_LADDER=1
run_arm al_goal_dm03 PARTNER_POOL_GATE=0 PARTNER_BUDGET_SCALE=6.0e-5 PARTNER_BUDGET_TAU=12 PARTNER_BUDGET_MIN=0.05 PARTNER_BUDGET_MAX=1.0 PARTNER_ANYTIME_LADDER=1 PARTNER_DIRECT_MIN=0.3 PARTNER_NREF=6
echo "=== chain done $(date '+%H:%M:%S') ==="
