#!/bin/bash
# Early-exit probe: how much wall does EARLY_EXIT return, at what quality cost,
# and does re-spending the returned time via a larger MAX beat control at
# iso-runtime? (design note docs/design/2026-08-04-early-exit-true-time-reduction.md §4)
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

run_arm ee_ctrl35   PARTNER_BUDGET_MAX=3.5
run_arm ee_on35     PARTNER_BUDGET_MAX=3.5 PARTNER_EARLY_EXIT=1
run_arm ee_on50     PARTNER_BUDGET_MAX=5.0 PARTNER_EARLY_EXIT=1
run_arm ee_on35_pg0 PARTNER_BUDGET_MAX=3.5 PARTNER_EARLY_EXIT=1 PARTNER_POOL_GATE=0
echo "=== chain done $(date '+%H:%M:%S') ==="
