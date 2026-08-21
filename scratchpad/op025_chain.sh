#!/bin/bash
# Operating-point refinement at the 0.2s-goal tier (MAX=0.25/MIN=0.15, gate=0):
#   pg025_rep2   : drift bracket for pg_max025
#   pg025_ee     : + EARLY_EXIT (converged workers return time)
#   pg025_noflow : FLOW_SLOTS=0 -- serial GPU head (~0.12s) is ~half the budget;
#                  does cutting it (doubling SA time) beat the flow seeds' value?
#   pg05_ee      : EE at the 0.5 tier
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

run_arm pg025_rep2   PARTNER_BUDGET_MAX=0.25 PARTNER_BUDGET_MIN=0.15 PARTNER_POOL_GATE=0
run_arm pg025_ee     PARTNER_BUDGET_MAX=0.25 PARTNER_BUDGET_MIN=0.15 PARTNER_POOL_GATE=0 PARTNER_EARLY_EXIT=1
run_arm pg025_noflow PARTNER_BUDGET_MAX=0.25 PARTNER_BUDGET_MIN=0.15 PARTNER_POOL_GATE=0 PARTNER_FLOW_SLOTS=0
run_arm pg05_ee      PARTNER_BUDGET_MAX=0.5 PARTNER_BUDGET_MIN=0.3 PARTNER_POOL_GATE=0 PARTNER_EARLY_EXIT=1
echo "=== chain done $(date '+%H:%M:%S') ==="
