#!/bin/bash
# alpha-interpolation fidelity probe.  PROBE ONLY -- every arm except `c`
# injects a ground-truth-INTERPOLATED layout and can never be promoted.
# Identical env to gt3_arm.sh (shipped 0.3s operating point).
# usage: alpha_arm.sh <arm:c|a00|a25|a50|a75|o> <rep> [dumpdir]
set -euo pipefail
ARM="$1"
REP="$2"
DUMP="${3:-}"
ROOT="/nashome/NVL4/vdalab/yyds-dev/Data-Driven-SoC-Floorplanning"
OUT="/tmp/claude-1100/-nashome-NVL4-vdalab-yyds-dev-Data-Driven-SoC-Floorplanning/c70131c1-0951-4c49-9b1e-f46a14d6ebfc/scratchpad"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$ROOT/FloorSet/iccad2026contest:$ROOT/FloorSet"
export DIRECT_CKPT="$ROOT/partner/checkpoints/direct_v2_cont/eval_step1p2M.pt"
export FLOW_CKPT="/nashome/NVL4/vdalab/yyds-dev/codex-worktrees/flow-matching-f1-f3/checkpoints/flow_matching_v1/final.pt"
# --- shipped 0.3s operating point (.env "0806 定案") -----------------------
export PARTNER_FLOW_SLOTS=10 PARTNER_FLOW_STEPS=8 PARTNER_FLOW_SOLVER=euler PARTNER_FLOW_ANTITHETIC=1
export VKILL_OFF=1 PARTNER_PRESCREEN_V=1 PARTNER_OVERSAMPLE=4 PARTNER_TAG_ANCHOR_EXTRA=3
export PARTNER_DIRECT_SOLVER=dpmpp PARTNER_REFINE_STALL_STOP=1
export PARTNER_SA_KERNEL=numba PARTNER_REFINE_KERNEL=numba PARTNER_FAST_SETUP=1
export PARTNER_POOL_GATE=0 PARTNER_BUDGET_TAU=12 PARTNER_BUDGET_MIN=0.05
export PARTNER_NREF=6 PARTNER_DIRECT_MIN=0.3 PARTNER_DDIM_STEPS=2
export PARTNER_BUDGET_SCALE=8.498e-5 PARTNER_BUDGET_MAX=1.22
export PARTNER_EDGE_SEAT_V2=1 PARTNER_FRAME_WPIN=1 PARTNER_COORD_POLISH=1
# --- probe arms -----------------------------------------------------------
unset PARTNER_ORACLE_PRED_FILE PARTNER_ORACLE_PRED_K PARTNER_PSEL_DUMP PARTNER_PSEL_TAG
case "$ARM" in
  c)   ;;                                                       # alpha = 0, no injection
  a00) export PARTNER_ORACLE_PRED_FILE="$OUT/alpha_a00.json" ;;  # alpha = 0, injected
  a25) export PARTNER_ORACLE_PRED_FILE="$OUT/alpha_a25.json" ;;
  a50) export PARTNER_ORACLE_PRED_FILE="$OUT/alpha_a50.json" ;;
  a75) export PARTNER_ORACLE_PRED_FILE="$OUT/alpha_a75.json" ;;
  o)   export PARTNER_ORACLE_PRED_FILE="$OUT/gr_layouts3.json" ;; # alpha = 1
  *)   echo "unknown arm $ARM" >&2; exit 2 ;;
esac
TAGSUF=""
if [ -n "$DUMP" ]; then
  export PARTNER_PSEL_DUMP="$DUMP"
  export PARTNER_PSEL_TAG="alpha_${ARM}_r${REP}"
  TAGSUF="_dump"
fi

echo "=== ALPHA arm=$ARM rep=$REP dump=${DUMP:-none} $(date +%H:%M:%S) ==="
cd "$ROOT/FloorSet/iccad2026contest" && \
uv run python "$ROOT/scripts/iccad2026_evaluate.py" \
  --data-path ../ --evaluate "$ROOT/partner/contest_optimizer.py" \
  --output "$OUT/eval_alpha_${ARM}_r${REP}${TAGSUF}.json" 2>&1 | tail -25
