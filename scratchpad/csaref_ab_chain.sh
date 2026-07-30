#!/bin/bash
# CSA coordinate-refine A/B: control (off) vs the two placements.
#   stall = in-loop at the median sweep's fixed point (competes with squeeze)
#   end   = terminal, carved out of the caller's span (squeeze already banked)
# Same 0729 base env as scratchpad/refstall_ab_chain.sh (dpmpp10 + DDIM10 +
# PARTNER_REFINE_STALL_STOP=1).  3 contemporaneous rounds; only same-round
# deltas are trusted.
ROOT="/nashome/NVL4/vdalab/yyds-dev/Data-Driven-SoC-Floorplanning"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$ROOT/FloorSet/iccad2026contest:$ROOT/FloorSet"
export DIRECT_CKPT="$ROOT/partner/checkpoints/direct_v2_cont/eval_step1p2M.pt"
export FLOW_CKPT="/nashome/NVL4/vdalab/yyds-dev/codex-worktrees/flow-matching-f1-f3/checkpoints/flow_matching_v1/final.pt"
export PARTNER_FLOW_SLOTS=10 PARTNER_FLOW_STEPS=8 PARTNER_FLOW_SOLVER=euler PARTNER_FLOW_ANTITHETIC=1
export VKILL_OFF=1 PARTNER_PRESCREEN_V=1 PARTNER_NREF=15 \
       PARTNER_OVERSAMPLE=4 PARTNER_TAG_ANCHOR_EXTRA=3
export PARTNER_BUDGET_MAX=3.5 PARTNER_DIRECT_MIN=2.0 PARTNER_DIRECT_SOLVER=dpmpp PARTNER_DDIM_STEPS=10
export PARTNER_REFINE_STALL_STOP=1
cd "$ROOT/FloorSet/iccad2026contest"

run_one () {   # $1 = cfg, $2 = round suffix
  unset PARTNER_CSA_REFINE PARTNER_CSA_WHERE
  case "$1" in
    stall) export PARTNER_CSA_REFINE=1 PARTNER_CSA_WHERE=stall ;;
    end)   export PARTNER_CSA_REFINE=1 PARTNER_CSA_WHERE=end ;;
  esac
  OUT="$ROOT/artifacts/partner_eval/csaref_$1$2.json"
  echo "=== $(date +%H:%M:%S) cfg=$1 round=$2 -> $OUT"
  uv run python "$ROOT/scripts/iccad2026_evaluate.py" \
    --data-path ../ --evaluate "$ROOT/partner/contest_optimizer.py" \
    --output "$OUT" 2>&1 | tail -4
}

for SUF in "" "_rep2" "_rep3"; do
  for CFG in control stall end; do
    run_one "$CFG" "$SUF"
  done
done
echo "=== ALL DONE $(date +%H:%M:%S)"
