#!/bin/bash
ROOT="/nashome/NVL4/vdalab/yyds-dev/Data-Driven-SoC-Floorplanning"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$ROOT/FloorSet/iccad2026contest:$ROOT/FloorSet"
export DIRECT_CKPT="$ROOT/partner/checkpoints/direct_v2_cont/eval_step1p2M.pt"
export FLOW_CKPT="/nashome/NVL4/vdalab/yyds-dev/codex-worktrees/flow-matching-f1-f3/checkpoints/flow_matching_v1/final.pt"
export PARTNER_FLOW_SLOTS=10 PARTNER_FLOW_STEPS=8 PARTNER_FLOW_SOLVER=euler PARTNER_FLOW_ANTITHETIC=1
export VKILL_OFF=1 PARTNER_PRESCREEN_V=1 PARTNER_NREF=15 PARTNER_OVERSAMPLE=4 PARTNER_TAG_ANCHOR_EXTRA=3
export PARTNER_BUDGET_MAX=3.5 PARTNER_DIRECT_MIN=2.0 PARTNER_DIRECT_SOLVER=dpmpp PARTNER_DDIM_STEPS=10
export PARTNER_REFINE_STALL_STOP=1
cd "$ROOT/FloorSet/iccad2026contest"
for REP in 1 2 3; do
for CFG in off on lowshare; do
  unset PARTNER_CSA_REFINE PARTNER_CSA_SHARE
  [ "$CFG" = on ] && export PARTNER_CSA_REFINE=1
  [ "$CFG" = lowshare ] && export PARTNER_CSA_REFINE=1 PARTNER_CSA_SHARE=0.02
  uv run python "$ROOT/scripts/iccad2026_evaluate.py" --data-path ../ \
    --evaluate "$ROOT/partner/contest_optimizer.py" --test-id 95 \
    --output "$ROOT/artifacts/partner_eval/csaprobe_${CFG}_$REP.json" >/dev/null 2>&1
  echo -n "rep$REP $CFG: "
  uv run python -c "
import json,sys
d=json.load(open('$ROOT/artifacts/partner_eval/csaprobe_${CFG}_$REP.json'))
r=d['test_results'][0]
ks=[k for k in r if 'gap' in k or 'viol' in k.lower() or k in ('cost_no_runtime','is_feasible')]
print(' '.join(f'{k}={r[k]:.4f}' if isinstance(r[k],float) else f'{k}={r[k]}' for k in sorted(ks)))
"
done
done
