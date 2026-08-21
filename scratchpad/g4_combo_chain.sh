#!/bin/bash
# Goal-tier round 4: compose the four independent G3 wins (st6/flow6/AL/trim)
# and probe st4.  Base G4 = RK + DM0.3 + NREF6 + gate0 + kernel + tau12.
ROOT="/nashome/NVL4/vdalab/yyds-dev/Data-Driven-SoC-Floorplanning"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$ROOT/FloorSet/iccad2026contest:$ROOT/FloorSet"

export DIRECT_CKPT="$ROOT/partner/checkpoints/direct_v2_cont/eval_step1p2M.pt"
export FLOW_CKPT="/nashome/NVL4/vdalab/yyds-dev/codex-worktrees/flow-matching-f1-f3/checkpoints/flow_matching_v1/final.pt"
export PARTNER_FLOW_SLOTS=10 PARTNER_FLOW_SOLVER=euler PARTNER_FLOW_ANTITHETIC=1
export VKILL_OFF=1 PARTNER_PRESCREEN_V=1 \
       PARTNER_OVERSAMPLE=4 PARTNER_TAG_ANCHOR_EXTRA=3
export PARTNER_DIRECT_SOLVER=dpmpp PARTNER_REFINE_STALL_STOP=1
export PARTNER_SA_KERNEL=numba PARTNER_POOL_GATE=0 PARTNER_REFINE_KERNEL=numba
export PARTNER_BUDGET_TAU=12 PARTNER_BUDGET_MIN=0.05 PARTNER_BUDGET_MAX=0.75
export PARTNER_NREF=6 PARTNER_DIRECT_MIN=0.3

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

run_arm g4_st6ctrl   PARTNER_BUDGET_SCALE=5.0e-5 PARTNER_DDIM_STEPS=6 PARTNER_FLOW_STEPS=8
run_arm g4_st4       PARTNER_BUDGET_SCALE=5.0e-5 PARTNER_DDIM_STEPS=4 PARTNER_FLOW_STEPS=8
run_arm g4_combo     PARTNER_BUDGET_SCALE=5.0e-5 PARTNER_DDIM_STEPS=6 PARTNER_FLOW_STEPS=6 PARTNER_ANYTIME_LADDER=1
run_arm g4_combo_r2  PARTNER_BUDGET_SCALE=5.0e-5 PARTNER_DDIM_STEPS=6 PARTNER_FLOW_STEPS=6 PARTNER_ANYTIME_LADDER=1
run_arm g4_combotrim PARTNER_BUDGET_SCALE=4.6e-5 PARTNER_DDIM_STEPS=6 PARTNER_FLOW_STEPS=6 PARTNER_ANYTIME_LADDER=1
echo "=== chain done $(date '+%H:%M:%S') ==="
