#!/bin/bash
# Chain EM: pessimistic slow-CPU emulation (budget table mid_m145 = budgets /1.45, gate thresholds scaled), v1 vs 300k EMA, official + v3, x2 alternating, package env otherwise.
cd /ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning
while pgrep -f "iccad2026_evaluate.py" >/dev/null; do sleep 20; done
S=scripts/gate/run_shadow.sh
TAB=$(cat artifacts/p0_newbox/budget_table_mid_m145.txt)
export CUDA_VISIBLE_DEVICES=3
CK=(DIRECT_OFF= "DIRECT_CKPT=$PWD/artifacts/icdc_topology/checkpoints_s2_20k/best.pt" PARTNER_DIRECT_SEAT_FIX=1 PARTNER_NREF=9 PARTNER_FLOW_SLOTS=10 PARTNER_REFINE_RES_FRAC=0.45 PARTNER_WALL_REPAIR=1 PARTNER_FLOW_WARM=1 PARTNER_EARLY_EXIT=1 PARTNER_REFINE_SECURE_FALLBACK=1 PARTNER_SEAT_R0_ADAPT=0 PARTNER_COORD_POLISH= PARTNER_FLOW_ANTITHETIC=1 "PARTNER_BUDGET_TABLE=$TAB")
V1=$PWD/submission/cadc1013/checkpoints/flow_matching_v1_final.pt
FT=$PWD/submission/cadc1013/checkpoints/flow_matching_ft0828_tailT24_300k_ema.pt
V3=$PWD/shadow_hidden/shadow_hidden_v3_share
bash $S emV1_r1_off $PWD/FloorSet "${CK[@]}" "FLOW_CKPT=$V1";  bash $S emV1_r1_v3 $V3 "${CK[@]}" "FLOW_CKPT=$V1"
bash $S emFT_r1_off $PWD/FloorSet "${CK[@]}" "FLOW_CKPT=$FT";  bash $S emFT_r1_v3 $V3 "${CK[@]}" "FLOW_CKPT=$FT"
bash $S emFT_r2_off $PWD/FloorSet "${CK[@]}" "FLOW_CKPT=$FT";  bash $S emFT_r2_v3 $V3 "${CK[@]}" "FLOW_CKPT=$FT"
bash $S emV1_r2_off $PWD/FloorSet "${CK[@]}" "FLOW_CKPT=$V1";  bash $S emV1_r2_v3 $V3 "${CK[@]}" "FLOW_CKPT=$V1"
echo CHAINEM_DONE
