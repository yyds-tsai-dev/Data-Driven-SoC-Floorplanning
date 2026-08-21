#!/bin/bash
# Exact-0.2 goal-point trim: floors put small/mid walls at ~0.13s, so
# avg<=0.20 needs tail avg <=0.46s.  b(n)=5e-5*e^(n/12) clamp[0.05,0.75]
# gives tail avg ~0.42 -> nominal sum ~19.1s.
#   gp_ctrl : tau12 MAX=1.0 family rep (drift bracket)
#   gp_020  : SCALE=5e-5 MAX=0.75 (the avg<=0.20 configuration)
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
export PARTNER_SA_KERNEL=numba PARTNER_POOL_GATE=0
export PARTNER_BUDGET_TAU=12 PARTNER_BUDGET_MIN=0.05

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

run_arm gp_ctrl PARTNER_BUDGET_SCALE=6.0e-5 PARTNER_BUDGET_MAX=1.0
run_arm gp_020  PARTNER_BUDGET_SCALE=5.0e-5 PARTNER_BUDGET_MAX=0.75
echo "=== chain done $(date '+%H:%M:%S') ==="
