#!/bin/bash
# Pool-floor scan: pool orchestration costs ~0.1-0.17s/case regardless of
# budget, and n<89 cases carry ~0.4-7% lambda weight -- route them through the
# cheap sequential path (gate>0) and pour the freed seconds into the tail
# (SCALE up).  All arms: kernel on, tau12-family shape, MIN=0.05.
#   pf_ctrl   : sh_tau12 rep (drift bracket)
#   pf_g01    : GATE=0.1  SCALE=7e-5 MAX=1.0  (n<89 sequential; tail ~0.55s avg)
#   pf_g02    : GATE=0.2  SCALE=7e-5 MAX=1.0  (n<97 sequential)
#   pf_g01max : GATE=0.1  SCALE=7e-5 MAX=1.3  (top cases up to 1.3s)
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
export PARTNER_SA_KERNEL=numba PARTNER_BUDGET_MIN=0.05 PARTNER_BUDGET_TAU=12

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

run_arm pf_ctrl   PARTNER_POOL_GATE=0   PARTNER_BUDGET_SCALE=6.0e-5 PARTNER_BUDGET_MAX=1.0
run_arm pf_g01    PARTNER_POOL_GATE=0.1 PARTNER_BUDGET_SCALE=7.0e-5 PARTNER_BUDGET_MAX=1.0
run_arm pf_g02    PARTNER_POOL_GATE=0.2 PARTNER_BUDGET_SCALE=7.0e-5 PARTNER_BUDGET_MAX=1.0
run_arm pf_g01max PARTNER_POOL_GATE=0.1 PARTNER_BUDGET_SCALE=7.0e-5 PARTNER_BUDGET_MAX=1.3
echo "=== chain done $(date '+%H:%M:%S') ==="
