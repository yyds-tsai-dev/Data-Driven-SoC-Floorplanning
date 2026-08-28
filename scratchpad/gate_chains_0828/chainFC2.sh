#!/bin/bash
# Chain FC2: 2 more reps of final-candidate env (polish headroom + anti0) vs package env, reverse order (New first).
cd /ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning
while pgrep -f "iccad2026_evaluate.py" >/dev/null; do sleep 20; done
G=scripts/gate/run_gate5.sh
TABM=$(cat artifacts/p0_newbox/budget_table_mid.txt)
export CUDA_VISIBLE_DEVICES=3
CK=(DIRECT_OFF= "DIRECT_CKPT=$PWD/artifacts/icdc_topology/checkpoints_s2_20k/best.pt" "FLOW_CKPT=$PWD/submission/cadc1013/checkpoints/flow_matching_v1_final.pt" PARTNER_DIRECT_SEAT_FIX=1 PARTNER_NREF=9 PARTNER_FLOW_SLOTS=10 PARTNER_REFINE_RES_FRAC=0.45 PARTNER_WALL_REPAIR=1 PARTNER_FLOW_WARM=1 PARTNER_EARLY_EXIT=1 PARTNER_REFINE_SECURE_FALLBACK=1 PARTNER_SEAT_R0_ADAPT=0 "PARTNER_BUDGET_TABLE=$TABM")
OLD=(PARTNER_COORD_POLISH= PARTNER_FLOW_ANTITHETIC=1)
NEW=(PARTNER_COORD_POLISH=1 PARTNER_COORD_POLISH_HEADROOM_S=0.6 PARTNER_FLOW_ANTITHETIC=0)
bash $G fcNew_r3 "${CK[@]}" "${NEW[@]}"
bash $G fcOld_r3 "${CK[@]}" "${OLD[@]}"
bash $G fcOld_r4 "${CK[@]}" "${OLD[@]}"
bash $G fcNew_r4 "${CK[@]}" "${NEW[@]}"
echo CHAINFC2_DONE
