#!/bin/bash
# Low-budget frontier chain (goal: map Q-time elasticity down to the 0.2s/case regime).
# Base env = 0729 定案三件套 (dpmpp10 + DDIM10 + REFINE_STALL_STOP) on MAX=3.5/DM2.0.
# Controls bracket the chain (rep1 first, rep2 last) for same-period drift.
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
      --output "$ROOT/artifacts/partner_eval/${name}.json" 2>&1 | tail -6
  )
  echo "=== $name end $(date '+%H:%M:%S') ==="
}

run_arm lbf_ctrl35_rep1 PARTNER_BUDGET_MAX=3.5
run_arm lbf_max20      PARTNER_BUDGET_MAX=2.0
run_arm lbf_max10      PARTNER_BUDGET_MAX=1.0
run_arm lbf_max05      PARTNER_BUDGET_MAX=0.5 PARTNER_BUDGET_MIN=0.3
run_arm lbf_max05_dm   PARTNER_BUDGET_MAX=0.5 PARTNER_BUDGET_MIN=0.3 PARTNER_DIRECT_MIN=0.05
run_arm lbf_max025     PARTNER_BUDGET_MAX=0.25 PARTNER_BUDGET_MIN=0.15
run_arm lbf_ctrl35_rep2 PARTNER_BUDGET_MAX=3.5
echo "=== chain done $(date '+%H:%M:%S') ==="
