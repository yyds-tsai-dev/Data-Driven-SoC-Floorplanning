#!/bin/bash
# Budget-table chain part 2: v5 + v6 suites, base/k20/k25 x2, flipped order.
cd /ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning
T=/ldaphome/yyds-tsai-dev/.claude/jobs/06af0e53/tmp
until grep -q CHAINRT_DONE $T/chainRT.log 2>/dev/null; do sleep 30; done
while pgrep -f "iccad2026_evaluate.py" >/dev/null; do sleep 20; done
export CUDA_VISIBLE_DEVICES=3
S=scripts/gate/run_shadow.sh
TABM=$(cat artifacts/p0_newbox/budget_table_mid.txt)
T20=$(cat scratchpad/rtaware/budget_table_k20.txt)
T25=$(cat scratchpad/rtaware/budget_table_k25.txt)
FT=$PWD/submission/cadc1013/checkpoints/flow_matching_ft0828_tailT24_300k_ema.pt
CK=(DIRECT_OFF= "DIRECT_CKPT=$PWD/artifacts/icdc_topology/checkpoints_s2_20k/best.pt" PARTNER_DIRECT_SEAT_FIX=1 PARTNER_NREF=9 PARTNER_FLOW_SLOTS=10 PARTNER_REFINE_RES_FRAC=0.45 PARTNER_WALL_REPAIR=1 PARTNER_FLOW_WARM=1 PARTNER_EARLY_EXIT=1 PARTNER_REFINE_SECURE_FALLBACK=1 PARTNER_SEAT_R0_ADAPT=0 PARTNER_COORD_POLISH= PARTNER_FLOW_ANTITHETIC=1 "FLOW_CKPT=$FT")
run(){ bash $S "$1" "$2" "${CK[@]}" "PARTNER_BUDGET_TABLE=$3"; }
for s in v5 v6; do
  D=$PWD/shadow_hidden/shadow_hidden_${s}_share
  run rtBase_${s}r1 $D "$TABM"; run rtK20_${s}r1 $D "$T20"; run rtK25_${s}r1 $D "$T25"
  run rtK25_${s}r2 $D "$T25"; run rtK20_${s}r2 $D "$T20"; run rtBase_${s}r2 $D "$TABM"
done
echo CHAINRT2_DONE
