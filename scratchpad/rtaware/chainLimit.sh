#!/bin/bash
# Diagnostic only: how low does the no-runtime score go with much larger budgets? (FT2 model, package env otherwise)
cd /ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning
while pgrep -f "iccad2026_evaluate.py" >/dev/null; do sleep 20; done
export CUDA_VISIBLE_DEVICES=3
S=scripts/gate/run_shadow.sh
FT2=$PWD/artifacts/flow_ft_0829/flow_ft0829_tailT12_lr1e-5_250k_ema.pt
CK=(DIRECT_OFF= "DIRECT_CKPT=$PWD/artifacts/icdc_topology/checkpoints_s2_20k/best.pt" PARTNER_DIRECT_SEAT_FIX=1 PARTNER_NREF=9 PARTNER_FLOW_SLOTS=10 PARTNER_REFINE_RES_FRAC=0.45 PARTNER_WALL_REPAIR=1 PARTNER_FLOW_WARM=1 PARTNER_EARLY_EXIT=1 PARTNER_REFINE_SECURE_FALLBACK=1 PARTNER_SEAT_R0_ADAPT=0 PARTNER_COORD_POLISH= PARTNER_FLOW_ANTITHETIC=1 "FLOW_CKPT=$FT2")
OFF=$PWD/FloorSet; V3=$PWD/shadow_hidden/shadow_hidden_v3_share
tab(){ cat scratchpad/rtaware/budget_table_x$1.txt; }
bash $S limX1 $OFF "${CK[@]}" "PARTNER_BUDGET_TABLE=$(cat artifacts/p0_newbox/budget_table_mid.txt)"
bash $S limX3 $OFF "${CK[@]}" "PARTNER_BUDGET_TABLE=$(tab 3)"
bash $S limX10 $OFF "${CK[@]}" "PARTNER_BUDGET_TABLE=$(tab 10)"
bash $S limX10_ee0 $OFF "${CK[@]}" "PARTNER_BUDGET_TABLE=$(tab 10)" PARTNER_EARLY_EXIT=0
bash $S limX10_wide $OFF "${CK[@]}" "PARTNER_BUDGET_TABLE=$(tab 10)" PARTNER_FLOW_SLOTS=20 PARTNER_NREF=18
bash $S limX30 $OFF "${CK[@]}" "PARTNER_BUDGET_TABLE=$(tab 30)"
bash $S limX10_v3 $V3 "${CK[@]}" "PARTNER_BUDGET_TABLE=$(tab 10)"
echo CHAINLIMIT_DONE
