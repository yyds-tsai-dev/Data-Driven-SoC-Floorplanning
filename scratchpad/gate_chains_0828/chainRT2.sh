#!/bin/bash
# Chain RT2: plain 300k (EMA-only) and n-routed prior (v1 below 95, ft300k EMA-only at/above 95) vs base, four suites, x2 alternating; all arms = package env (md5 163b8854 config: polish off, antithetic 1).
cd /ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning
T=/ldaphome/yyds-tsai-dev/.claude/jobs/06af0e53/tmp
until grep -q CHAINFC2_DONE $T/chainFC2.log 2>/dev/null; do sleep 30; done
while pgrep -f "iccad2026_evaluate.py" >/dev/null; do sleep 20; done
SRC=$PWD/artifacts/flow_ft_0828/flow_ft0828_tailT24_lr2e-5_step300k.pt
TAIL=$PWD/artifacts/flow_ft_0828/flow_ft0828_tailT24_lr2e-5_step300k_ema.pt
[ -f $TAIL ] || uv run python artifacts/flow_ft_0828/export_ema_only.py $SRC $TAIL || { echo EMA_EXPORT_FAIL; exit 1; }
G=scripts/gate/run_gate5.sh
TABM=$(cat artifacts/p0_newbox/budget_table_mid.txt)
export CUDA_VISIBLE_DEVICES=3
CK=(DIRECT_OFF= "DIRECT_CKPT=$PWD/artifacts/icdc_topology/checkpoints_s2_20k/best.pt" PARTNER_DIRECT_SEAT_FIX=1 PARTNER_NREF=9 PARTNER_FLOW_SLOTS=10 PARTNER_REFINE_RES_FRAC=0.45 PARTNER_WALL_REPAIR=1 PARTNER_FLOW_WARM=1 PARTNER_EARLY_EXIT=1 PARTNER_REFINE_SECURE_FALLBACK=1 PARTNER_SEAT_R0_ADAPT=0 PARTNER_COORD_POLISH= PARTNER_FLOW_ANTITHETIC=1 "PARTNER_BUDGET_TABLE=$TABM")
V1=$PWD/submission/cadc1013/checkpoints/flow_matching_v1_final.pt
bash $G rt2Base_r1 "${CK[@]}" "FLOW_CKPT=$V1"
bash $G pl300_r1 "${CK[@]}" "FLOW_CKPT=$TAIL"
bash $G rt300N95_r1 "${CK[@]}" "FLOW_CKPT=$V1" "FLOW_CKPT_TAIL=$TAIL" PARTNER_FLOW_TAIL_MIN_N=95
bash $G rt300N95_r2 "${CK[@]}" "FLOW_CKPT=$V1" "FLOW_CKPT_TAIL=$TAIL" PARTNER_FLOW_TAIL_MIN_N=95
bash $G pl300_r2 "${CK[@]}" "FLOW_CKPT=$TAIL"
bash $G rt2Base_r2 "${CK[@]}" "FLOW_CKPT=$V1"
echo CHAINRT2_DONE
