#!/bin/bash
# Cherry-pick batch 1 A/B: control vs +PARTNER_FASTSA_TEMP vs +PARTNER_SA_STALL_STOP,
# all stacked on the current dpmpp10 baseline env (see scratchpad/dpmpp10_ab_rep2.sh).
# 3 contemporaneous rounds; only same-round deltas are trusted (GPU shared with flow v3).
ROOT="/nashome/NVL4/vdalab/yyds-dev/Data-Driven-SoC-Floorplanning"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$ROOT/FloorSet/iccad2026contest:$ROOT/FloorSet"
export DIRECT_CKPT="$ROOT/partner/checkpoints/direct_v2_cont/eval_step1p2M.pt"
export FLOW_CKPT="/nashome/NVL4/vdalab/yyds-dev/codex-worktrees/flow-matching-f1-f3/checkpoints/flow_matching_v1/final.pt"
export PARTNER_FLOW_SLOTS=10 PARTNER_FLOW_STEPS=8 PARTNER_FLOW_SOLVER=euler PARTNER_FLOW_ANTITHETIC=1
export VKILL_OFF=1 PARTNER_PRESCREEN_V=1 PARTNER_NREF=15 \
       PARTNER_OVERSAMPLE=4 PARTNER_TAG_ANCHOR_EXTRA=3
export PARTNER_BUDGET_MAX=3.5 PARTNER_DIRECT_MIN=2.0 PARTNER_DIRECT_SOLVER=dpmpp PARTNER_DDIM_STEPS=10
cd "$ROOT/FloorSet/iccad2026contest"

run_one () {   # $1 = cfg name, $2 = output suffix
  unset PARTNER_FASTSA_TEMP PARTNER_SA_STALL_STOP
  case "$1" in
    fastsa)    export PARTNER_FASTSA_TEMP=1 ;;
    stallstop) export PARTNER_SA_STALL_STOP=1 ;;
  esac
  OUT="$ROOT/artifacts/partner_eval/cherry1_$1$2.json"
  echo "=== $(date +%H:%M:%S) cfg=$1 round=$2 -> $OUT"
  uv run python "$ROOT/scripts/iccad2026_evaluate.py" \
    --data-path ../ --evaluate "$ROOT/partner/my_opt_claude.py" \
    --output "$OUT" 2>&1 | tail -4
}

for SUF in "" "_rep2" "_rep3"; do
  for CFG in control fastsa stallstop; do
    run_one "$CFG" "$SUF"
  done
done
echo "=== ALL DONE $(date +%H:%M:%S)"
