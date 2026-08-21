#!/bin/bash
# Budget-curve shape scan under the avg<=0.2s constraint: lambda ~ e^(n/12)
# is tail-dominated, so starve small cases (MIN=0.05) and pour the savings
# into the tail.  All arms: kernel + gate0.  Nominal sums ~17-18s (avg ~0.18)
# leaving slack for serial-head overflow on starved cases.
#   sh_flat025 : flat MIN=0.15/MAX=0.25 bracket (k_on025 config rep)
#   sh_tau8    : b(n)=9.3e-7*e^(n/8)   clamp[0.05,0.8]  (b(100)=0.25, n>=109: 0.8)
#   sh_tau10   : b(n)=1.54e-6*e^(n/10) clamp[0.05,1.0]  (b(110)=0.68, n>=112: 1.0)
#   sh_tau12   : b(n)=6.0e-5*e^(n/12)  clamp[0.05,1.0]  (matches lambda exponent)
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
export PARTNER_POOL_GATE=0 PARTNER_SA_KERNEL=numba

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

run_arm sh_flat025 PARTNER_BUDGET_MAX=0.25 PARTNER_BUDGET_MIN=0.15
run_arm sh_tau8    PARTNER_BUDGET_SCALE=9.3e-7 PARTNER_BUDGET_TAU=8 PARTNER_BUDGET_MIN=0.05 PARTNER_BUDGET_MAX=0.8
run_arm sh_tau10   PARTNER_BUDGET_SCALE=1.54e-6 PARTNER_BUDGET_TAU=10 PARTNER_BUDGET_MIN=0.05 PARTNER_BUDGET_MAX=1.0
run_arm sh_tau12   PARTNER_BUDGET_SCALE=6.0e-5 PARTNER_BUDGET_TAU=12 PARTNER_BUDGET_MIN=0.05 PARTNER_BUDGET_MAX=1.0
echo "=== chain done $(date '+%H:%M:%S') ==="
