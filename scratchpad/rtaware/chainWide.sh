#!/bin/bash
# x1 budget, FT2 model, package env: candidate-pool width test (FLOW_SLOTS/NREF) — official x2 paired, reversed.
cd /ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning
T=/ldaphome/yyds-tsai-dev/.claude/jobs/06af0e53/tmp
until grep -q CHAINLIMIT_DONE $T/chainLimit.log 2>/dev/null; do sleep 30; done
while pgrep -f "iccad2026_evaluate.py" >/dev/null; do sleep 20; done
export CUDA_VISIBLE_DEVICES=3
S=scripts/gate/run_shadow.sh
FT2=$PWD/artifacts/flow_ft_0829/flow_ft0829_tailT12_lr1e-5_250k_ema.pt
TABM=$(cat artifacts/p0_newbox/budget_table_mid.txt)
CK=(DIRECT_OFF= "DIRECT_CKPT=$PWD/artifacts/icdc_topology/checkpoints_s2_20k/best.pt" PARTNER_DIRECT_SEAT_FIX=1 PARTNER_REFINE_RES_FRAC=0.45 PARTNER_WALL_REPAIR=1 PARTNER_FLOW_WARM=1 PARTNER_EARLY_EXIT=1 PARTNER_REFINE_SECURE_FALLBACK=1 PARTNER_SEAT_R0_ADAPT=0 PARTNER_COORD_POLISH= PARTNER_FLOW_ANTITHETIC=1 "FLOW_CKPT=$FT2" "PARTNER_BUDGET_TABLE=$TABM")
OFF=$PWD/FloorSet
bash $S wdBase_r1 $OFF "${CK[@]}" PARTNER_NREF=9 PARTNER_FLOW_SLOTS=10
bash $S wd20_r1 $OFF "${CK[@]}" PARTNER_NREF=18 PARTNER_FLOW_SLOTS=20
bash $S wd16_r1 $OFF "${CK[@]}" PARTNER_NREF=12 PARTNER_FLOW_SLOTS=16
bash $S wd16_r2 $OFF "${CK[@]}" PARTNER_NREF=12 PARTNER_FLOW_SLOTS=16
bash $S wd20_r2 $OFF "${CK[@]}" PARTNER_NREF=18 PARTNER_FLOW_SLOTS=20
bash $S wdBase_r2 $OFF "${CK[@]}" PARTNER_NREF=9 PARTNER_FLOW_SLOTS=10
echo CHAINWIDE_DONE
