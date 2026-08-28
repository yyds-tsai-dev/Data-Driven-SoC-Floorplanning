#!/bin/bash
# Chain PO: official-only, plain 300k (EMA-only) vs v1, x3 each, alternating, package env.
cd /ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning
T=/ldaphome/yyds-tsai-dev/.claude/jobs/06af0e53/tmp
until grep -q CHAINFD_DONE $T/chainFD.log 2>/dev/null; do sleep 30; done
while pgrep -f "iccad2026_evaluate.py" >/dev/null; do sleep 20; done
S=scripts/gate/run_shadow.sh
TABM=$(cat artifacts/p0_newbox/budget_table_mid.txt)
export CUDA_VISIBLE_DEVICES=3
CK=(DIRECT_OFF= "DIRECT_CKPT=$PWD/artifacts/icdc_topology/checkpoints_s2_20k/best.pt" PARTNER_DIRECT_SEAT_FIX=1 PARTNER_NREF=9 PARTNER_FLOW_SLOTS=10 PARTNER_REFINE_RES_FRAC=0.45 PARTNER_WALL_REPAIR=1 PARTNER_FLOW_WARM=1 PARTNER_EARLY_EXIT=1 PARTNER_REFINE_SECURE_FALLBACK=1 PARTNER_SEAT_R0_ADAPT=0 PARTNER_COORD_POLISH= PARTNER_FLOW_ANTITHETIC=1 "PARTNER_BUDGET_TABLE=$TABM")
V1=$PWD/submission/cadc1013/checkpoints/flow_matching_v1_final.pt
TAIL=$PWD/artifacts/flow_ft_0828/flow_ft0828_tailT24_lr2e-5_step300k_ema.pt
bash $S poV1_r3_off $PWD/FloorSet "${CK[@]}" "FLOW_CKPT=$V1"
bash $S po300_r3_off $PWD/FloorSet "${CK[@]}" "FLOW_CKPT=$TAIL"
bash $S po300_r4_off $PWD/FloorSet "${CK[@]}" "FLOW_CKPT=$TAIL"
bash $S poV1_r4_off $PWD/FloorSet "${CK[@]}" "FLOW_CKPT=$V1"
bash $S poV1_r5_off $PWD/FloorSet "${CK[@]}" "FLOW_CKPT=$V1"
bash $S po300_r5_off $PWD/FloorSet "${CK[@]}" "FLOW_CKPT=$TAIL"
echo CHAINPO_DONE
