#!/bin/bash
# Tail-budget calibration chain: package env (mid table) vs k20 / k25 tail-calibrated tables.
# official ×2 interleaved, order flipped in rep 2; then v3 ×2 same pattern.
cd /ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning
while pgrep -f "iccad2026_evaluate.py" >/dev/null; do sleep 20; done
export CUDA_VISIBLE_DEVICES=3
S=scripts/gate/run_shadow.sh
TABM=$(cat artifacts/p0_newbox/budget_table_mid.txt)
T20=$(cat scratchpad/rtaware/budget_table_k20.txt)
T25=$(cat scratchpad/rtaware/budget_table_k25.txt)
FT=$PWD/submission/cadc1013/checkpoints/flow_matching_ft0828_tailT24_300k_ema.pt
CK=(DIRECT_OFF= "DIRECT_CKPT=$PWD/artifacts/icdc_topology/checkpoints_s2_20k/best.pt" PARTNER_DIRECT_SEAT_FIX=1 PARTNER_NREF=9 PARTNER_FLOW_SLOTS=10 PARTNER_REFINE_RES_FRAC=0.45 PARTNER_WALL_REPAIR=1 PARTNER_FLOW_WARM=1 PARTNER_EARLY_EXIT=1 PARTNER_REFINE_SECURE_FALLBACK=1 PARTNER_SEAT_R0_ADAPT=0 PARTNER_COORD_POLISH= PARTNER_FLOW_ANTITHETIC=1 "FLOW_CKPT=$FT")
OFF=$PWD/FloorSet; V3=$PWD/shadow_hidden/shadow_hidden_v3_share
run(){ bash $S "$1" "$2" "${CK[@]}" "PARTNER_BUDGET_TABLE=$3"; }
run rtBase_r1 $OFF "$TABM"; run rtK20_r1 $OFF "$T20"; run rtK25_r1 $OFF "$T25"
run rtK25_r2 $OFF "$T25"; run rtK20_r2 $OFF "$T20"; run rtBase_r2 $OFF "$TABM"
echo CHAINRT_OFF_DONE
run rtBase_v3r1 $V3 "$TABM"; run rtK20_v3r1 $V3 "$T20"; run rtK25_v3r1 $V3 "$T25"
run rtK25_v3r2 $V3 "$T25"; run rtK20_v3r2 $V3 "$T20"; run rtBase_v3r2 $V3 "$TABM"
echo CHAINRT_DONE
