#!/bin/bash
# Chain RT1: n-routed prior (v1 below 95, ft90k at/above 95) vs base, four suites, x2 alternating; both arms on the new shipping base (polish headroom on).
cd /ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning
T=/ldaphome/yyds-tsai-dev/.claude/jobs/06af0e53/tmp
until grep -q CHAINN2_DONE $T/chainN2.log 2>/dev/null; do sleep 30; done
while pgrep -f "iccad2026_evaluate.py" >/dev/null; do sleep 20; done
G=scripts/gate/run_gate5.sh
TABM=$(cat artifacts/p0_newbox/budget_table_mid.txt)
export CUDA_VISIBLE_DEVICES=3
CK=(DIRECT_OFF= "DIRECT_CKPT=$PWD/artifacts/icdc_topology/checkpoints_s2_20k/best.pt" PARTNER_DIRECT_SEAT_FIX=1 PARTNER_NREF=9 PARTNER_FLOW_SLOTS=10 PARTNER_REFINE_RES_FRAC=0.45 PARTNER_WALL_REPAIR=1 PARTNER_FLOW_WARM=1 PARTNER_EARLY_EXIT=1 PARTNER_REFINE_SECURE_FALLBACK=1 PARTNER_SEAT_R0_ADAPT=0 PARTNER_COORD_POLISH=1 PARTNER_COORD_POLISH_HEADROOM_S=0.6 "PARTNER_BUDGET_TABLE=$TABM")
V1=$PWD/submission/cadc1013/checkpoints/flow_matching_v1_final.pt
TAIL=$PWD/artifacts/flow_ft_0828/flow_ft0828_tailT24_lr2e-5_step90k.pt
bash $G rtBase_r1 "${CK[@]}" "FLOW_CKPT=$V1"
bash $G rt90N95_r1 "${CK[@]}" "FLOW_CKPT=$V1" "FLOW_CKPT_TAIL=$TAIL" PARTNER_FLOW_TAIL_MIN_N=95
bash $G rt90N95_r2 "${CK[@]}" "FLOW_CKPT=$V1" "FLOW_CKPT_TAIL=$TAIL" PARTNER_FLOW_TAIL_MIN_N=95
bash $G rtBase_r2 "${CK[@]}" "FLOW_CKPT=$V1"
echo CHAINRT1_DONE
