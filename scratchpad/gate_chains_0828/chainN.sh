#!/bin/bash
cd /ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning
T=/ldaphome/yyds-tsai-dev/.claude/jobs/06af0e53/tmp
until grep -q CHAINP3_DONE $T/chainP3.log 2>/dev/null; do sleep 30; done
while pgrep -f "iccad2026_evaluate.py" >/dev/null; do sleep 20; done
G=$T/run_gate5.sh
TABM=$(cat artifacts/p0_newbox/budget_table_mid.txt); TAB108=$(cat artifacts/p0_newbox/budget_table_mid108.txt)
export CUDA_VISIBLE_DEVICES=3
CK=(DIRECT_OFF= "DIRECT_CKPT=$PWD/artifacts/icdc_topology/checkpoints_s2_20k/best.pt" "FLOW_CKPT=$PWD/submission/cadc1013/checkpoints/flow_matching_v1_final.pt" PARTNER_DIRECT_SEAT_FIX=1 PARTNER_NREF=9 PARTNER_FLOW_SLOTS=10 PARTNER_REFINE_RES_FRAC=0.45 PARTNER_WALL_REPAIR=1 PARTNER_FLOW_WARM=1 PARTNER_EARLY_EXIT=1 PARTNER_REFINE_SECURE_FALLBACK=1 PARTNER_SEAT_R0_ADAPT=0 PARTNER_COORD_POLISH=)
bash $G nBase_r1 "PARTNER_BUDGET_TABLE=$TABM" "${CK[@]}"
bash $G nHeun_r1 "PARTNER_BUDGET_TABLE=$TABM" "${CK[@]}" PARTNER_FLOW_SOLVER=heun
bash $G nT108_r1 "PARTNER_BUDGET_TABLE=$TAB108" "${CK[@]}"
bash $G nAnti0_r1 "PARTNER_BUDGET_TABLE=$TABM" "${CK[@]}" PARTNER_FLOW_ANTITHETIC=0
bash $G nAnti0_r2 "PARTNER_BUDGET_TABLE=$TABM" "${CK[@]}" PARTNER_FLOW_ANTITHETIC=0
bash $G nT108_r2 "PARTNER_BUDGET_TABLE=$TAB108" "${CK[@]}"
bash $G nHeun_r2 "PARTNER_BUDGET_TABLE=$TABM" "${CK[@]}" PARTNER_FLOW_SOLVER=heun
bash $G nBase_r2 "PARTNER_BUDGET_TABLE=$TABM" "${CK[@]}"
echo CHAINN_DONE
