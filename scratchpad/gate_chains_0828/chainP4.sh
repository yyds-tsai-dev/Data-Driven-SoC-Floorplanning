#!/bin/bash
# Chain P4: polish HEADROOM 0.6 s vs base, reverse arm order (Head first), x2 — pooled with P3 for the 4-rep verdict.
cd /ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning
T=/ldaphome/yyds-tsai-dev/.claude/jobs/06af0e53/tmp
until grep -q CHAINFT_DONE $T/chainFT.log 2>/dev/null; do sleep 30; done
while pgrep -f "iccad2026_evaluate.py" >/dev/null; do sleep 20; done
G=scripts/gate/run_gate5.sh
TABM=$(cat artifacts/p0_newbox/budget_table_mid.txt)
export CUDA_VISIBLE_DEVICES=3
CK=(DIRECT_OFF= "DIRECT_CKPT=$PWD/artifacts/icdc_topology/checkpoints_s2_20k/best.pt" "FLOW_CKPT=$PWD/submission/cadc1013/checkpoints/flow_matching_v1_final.pt" PARTNER_DIRECT_SEAT_FIX=1 PARTNER_NREF=9 PARTNER_FLOW_SLOTS=10 PARTNER_REFINE_RES_FRAC=0.45 PARTNER_WALL_REPAIR=1 PARTNER_FLOW_WARM=1 PARTNER_EARLY_EXIT=1 PARTNER_REFINE_SECURE_FALLBACK=1 PARTNER_SEAT_R0_ADAPT=0 "PARTNER_BUDGET_TABLE=$TABM")
bash $G p3Head_r3 "${CK[@]}" PARTNER_COORD_POLISH=1 PARTNER_COORD_POLISH_HEADROOM_S=0.6
bash $G p3Base_r3 "${CK[@]}" PARTNER_COORD_POLISH=
bash $G p3Base_r4 "${CK[@]}" PARTNER_COORD_POLISH=
bash $G p3Head_r4 "${CK[@]}" PARTNER_COORD_POLISH=1 PARTNER_COORD_POLISH_HEADROOM_S=0.6
echo CHAINP4_DONE
