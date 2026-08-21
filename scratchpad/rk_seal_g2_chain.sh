#!/bin/bash
# Seal RK@3.5 promotion (rep3 pair) + tune the goal-tier direct resurrection:
#   g2_rep2    : rk_goal_dm03 rep (drift)
#   g2_dm02/05 : DIRECT_MIN sweep (which tail cases should sample?)
#   g2_nref10  : more refine slots now that rungs are cheap
#   g2_steps8  : dpmpp 10->8 (GPU 0.16->0.13s matters at 0.44s budgets)
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

G2="PARTNER_SA_KERNEL=numba PARTNER_POOL_GATE=0 PARTNER_BUDGET_SCALE=5.0e-5 PARTNER_BUDGET_TAU=12 PARTNER_BUDGET_MIN=0.05 PARTNER_BUDGET_MAX=0.75 PARTNER_REFINE_KERNEL=numba PARTNER_NREF=6"

run_arm rk_ctrl35_r3 PARTNER_BUDGET_MAX=3.5
run_arm rk_on35_r3   PARTNER_BUDGET_MAX=3.5 PARTNER_REFINE_KERNEL=numba
run_arm g2_rep2      $G2 PARTNER_DIRECT_MIN=0.3
run_arm g2_dm02      $G2 PARTNER_DIRECT_MIN=0.2
run_arm g2_dm05      $G2 PARTNER_DIRECT_MIN=0.5
run_arm g2_nref10    $G2 PARTNER_DIRECT_MIN=0.3 PARTNER_NREF=10
run_arm g2_steps8    $G2 PARTNER_DIRECT_MIN=0.3 PARTNER_DDIM_STEPS=8
echo "=== chain done $(date '+%H:%M:%S') ==="
