#!/bin/bash
# Ladder hypothesis probe: is the residual +0.16 cliff (3.5->2.0, pool open)
# the direct-refine ladder's contribution vanishing below ~3s?
#   ctrl35_dmoff : MAX=3.5, direct OFF  -> if ~1.29 (=pg_max20), hypothesis holds
#   pg2_dmoff    : MAX=2.0 gate=0, direct OFF -> ladder's net value at 2.0
#   pg2_nref6    : MAX=2.0 gate=0, NREF=6     -> slot rebalance at 2.0
#   pg2_rep2     : MAX=2.0 gate=0 rerun       -> drift bracket
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

run_arm ctrl35_dmoff  PARTNER_BUDGET_MAX=3.5 PARTNER_DIRECT_MIN=99.0
run_arm pg2_dmoff     PARTNER_BUDGET_MAX=2.0 PARTNER_POOL_GATE=0 PARTNER_DIRECT_MIN=99.0
run_arm pg2_nref6     PARTNER_BUDGET_MAX=2.0 PARTNER_POOL_GATE=0 PARTNER_NREF=6
run_arm pg2_rep2      PARTNER_BUDGET_MAX=2.0 PARTNER_POOL_GATE=0
echo "=== chain done $(date '+%H:%M:%S') ==="
