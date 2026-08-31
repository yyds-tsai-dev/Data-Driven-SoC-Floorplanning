#!/bin/bash
# Reproduce the legal-guard firing seen on shadow v3 tid 85 at x10 budget: guard OFF so the evaluator reports which hard check fails.
cd /ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning
T=/ldaphome/yyds-tsai-dev/.claude/jobs/06af0e53/tmp
until grep -q CHAINWIDE_DONE $T/chainWide.log 2>/dev/null; do sleep 30; done
while pgrep -f "iccad2026_evaluate.py" >/dev/null; do sleep 20; done
export CUDA_VISIBLE_DEVICES=3
S=scripts/gate/run_shadow.sh
FT2=$PWD/artifacts/flow_ft_0829/flow_ft0829_tailT12_lr1e-5_250k_ema.pt
CK=(DIRECT_OFF= "DIRECT_CKPT=$PWD/artifacts/icdc_topology/checkpoints_s2_20k/best.pt" PARTNER_DIRECT_SEAT_FIX=1 PARTNER_NREF=9 PARTNER_FLOW_SLOTS=10 PARTNER_REFINE_RES_FRAC=0.45 PARTNER_WALL_REPAIR=1 PARTNER_FLOW_WARM=1 PARTNER_EARLY_EXIT=1 PARTNER_REFINE_SECURE_FALLBACK=1 PARTNER_SEAT_R0_ADAPT=0 PARTNER_COORD_POLISH= PARTNER_FLOW_ANTITHETIC=1 "FLOW_CKPT=$FT2")
bash $S guardOff_x10_v3 $PWD/shadow_hidden/shadow_hidden_v3_share "${CK[@]}" "PARTNER_BUDGET_TABLE=$(cat scratchpad/rtaware/budget_table_x10.txt)" PARTNER_FINAL_LEGAL_GUARD=0
echo CHAINGUARD_DONE
