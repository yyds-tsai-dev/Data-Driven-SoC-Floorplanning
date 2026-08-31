#!/bin/bash
# alpha_1 (synthetic pin-shift shadow) on the three shipping candidates: P=0828b env, B'=FT2+s16, D'=FT2+k20+s16. x2 interleaved, reversed.
cd /ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning
while pgrep -f "iccad2026_evaluate.py" >/dev/null; do sleep 20; done
export CUDA_VISIBLE_DEVICES=3
S=scripts/gate/run_shadow.sh; A1=$PWD/shadow_hidden/alpha_1/alpha_1
FT1=$PWD/submission/cadc1013/checkpoints/flow_matching_ft0828_tailT24_300k_ema.pt
FT2=$PWD/artifacts/flow_ft_0829/flow_ft0829_tailT12_lr1e-5_250k_ema.pt
TABM=$(cat artifacts/p0_newbox/budget_table_mid.txt); T20=$(cat scratchpad/rtaware/budget_table_k20.txt)
CK=(DIRECT_OFF= "DIRECT_CKPT=$PWD/artifacts/icdc_topology/checkpoints_s2_20k/best.pt" PARTNER_DIRECT_SEAT_FIX=1 PARTNER_REFINE_RES_FRAC=0.45 PARTNER_WALL_REPAIR=1 PARTNER_FLOW_WARM=1 PARTNER_EARLY_EXIT=1 PARTNER_REFINE_SECURE_FALLBACK=1 PARTNER_SEAT_R0_ADAPT=0 PARTNER_COORD_POLISH= PARTNER_FLOW_ANTITHETIC=1)
P(){ bash $S a1P_r$1 $A1 "${CK[@]}" "FLOW_CKPT=$FT1" "PARTNER_BUDGET_TABLE=$TABM" PARTNER_NREF=9 PARTNER_FLOW_SLOTS=10; }
B(){ bash $S a1B_r$1 $A1 "${CK[@]}" "FLOW_CKPT=$FT2" "PARTNER_BUDGET_TABLE=$TABM" PARTNER_NREF=12 PARTNER_FLOW_SLOTS=16; }
D(){ bash $S a1D_r$1 $A1 "${CK[@]}" "FLOW_CKPT=$FT2" "PARTNER_BUDGET_TABLE=$T20" PARTNER_NREF=12 PARTNER_FLOW_SLOTS=16; }
P 1; B 1; D 1; D 2; B 2; P 2
echo CHAINA1_DONE
