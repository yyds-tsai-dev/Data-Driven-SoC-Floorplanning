#!/bin/bash
# CSA coordinate-refine smoke: 2 validation cases (one n>100), on/off.
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
for TID in 5 95; do
  for CFG in off on; do
    unset PARTNER_CSA_REFINE
    [ "$CFG" = on ] && export PARTNER_CSA_REFINE=1
    echo "--- test_id=$TID csa=$CFG ---"
    uv run python "$ROOT/scripts/iccad2026_evaluate.py" \
      --data-path ../ --evaluate "$ROOT/partner/contest_optimizer.py" \
      --test-id "$TID" 2>&1 | grep -Ei "feasib|cost|score|runtime|Traceback|Error" | head -12
  done
done
