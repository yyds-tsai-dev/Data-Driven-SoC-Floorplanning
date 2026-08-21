#!/bin/bash
# Pool-gate frontier: rerun the low-budget frontier with PARTNER_POOL_GATE=0
# (parallel portfolio + ML channels engage at every budget).  Hypothesis: the
# +0.22 cliff at MAX<=3.0 was the hard `budget > 3.0` pool gate; with it open
# the frontier should flatten toward the 24s->3.5s elasticity (~+0.01/doubling).
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

# sanity: gate default must reproduce control (bit-exact patch check)
run_arm pg_ctrl35        PARTNER_BUDGET_MAX=3.5
run_arm pg_max20         PARTNER_BUDGET_MAX=2.0 PARTNER_POOL_GATE=0
run_arm pg_max10         PARTNER_BUDGET_MAX=1.0 PARTNER_POOL_GATE=0
run_arm pg_max05         PARTNER_BUDGET_MAX=0.5 PARTNER_BUDGET_MIN=0.3 PARTNER_POOL_GATE=0
run_arm pg_max025        PARTNER_BUDGET_MAX=0.25 PARTNER_BUDGET_MIN=0.15 PARTNER_POOL_GATE=0
# direct-rescue retest, now that the channel can actually fire below 3s
run_arm pg_max05_dm      PARTNER_BUDGET_MAX=0.5 PARTNER_BUDGET_MIN=0.3 PARTNER_POOL_GATE=0 PARTNER_DIRECT_MIN=0.05
run_arm pg_max35_full    PARTNER_BUDGET_MAX=3.5 PARTNER_POOL_GATE=0
echo "=== chain done $(date '+%H:%M:%S') ==="
