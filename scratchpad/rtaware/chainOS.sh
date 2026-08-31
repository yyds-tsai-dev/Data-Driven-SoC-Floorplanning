#!/bin/bash
# Seed-quality knobs at x1 on top of B' (FT2 + s16): OS = oversample x3 with prescreen (KS_CAP 48, MIN_REM 0); ST = FLOW_STEPS 16. Official x2 paired, reversed.
cd /ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning
T=/ldaphome/yyds-tsai-dev/.claude/jobs/06af0e53/tmp
until grep -q CHAINS24_DONE $T/chainS24.log 2>/dev/null; do sleep 30; done
while pgrep -f "iccad2026_evaluate.py" >/dev/null; do sleep 20; done
export CUDA_VISIBLE_DEVICES=3
S=scripts/gate/run_shadow.sh; OFF=$PWD/FloorSet
FT2=$PWD/artifacts/flow_ft_0829/flow_ft0829_tailT12_lr1e-5_250k_ema.pt
TABM=$(cat artifacts/p0_newbox/budget_table_mid.txt)
CK=(DIRECT_OFF= "DIRECT_CKPT=$PWD/artifacts/icdc_topology/checkpoints_s2_20k/best.pt" PARTNER_DIRECT_SEAT_FIX=1 PARTNER_REFINE_RES_FRAC=0.45 PARTNER_WALL_REPAIR=1 PARTNER_FLOW_WARM=1 PARTNER_EARLY_EXIT=1 PARTNER_REFINE_SECURE_FALLBACK=1 PARTNER_SEAT_R0_ADAPT=0 PARTNER_COORD_POLISH= PARTNER_FLOW_ANTITHETIC=1 "FLOW_CKPT=$FT2" "PARTNER_BUDGET_TABLE=$TABM" PARTNER_NREF=12 PARTNER_FLOW_SLOTS=16)
bash $S osBase_r1 $OFF "${CK[@]}"
bash $S osOS3_r1 $OFF "${CK[@]}" PARTNER_OVERSAMPLE=3 PARTNER_KS_CAP=48 PARTNER_OVERSAMPLE_MIN_REM=0
bash $S osST16_r1 $OFF "${CK[@]}" PARTNER_FLOW_STEPS=16
bash $S osST16_r2 $OFF "${CK[@]}" PARTNER_FLOW_STEPS=16
bash $S osOS3_r2 $OFF "${CK[@]}" PARTNER_OVERSAMPLE=3 PARTNER_KS_CAP=48 PARTNER_OVERSAMPLE_MIN_REM=0
bash $S osBase_r2 $OFF "${CK[@]}"
for c in osOS3 osST16; do echo "== $c vs base"; uv run python scripts/gate/analyze_pairs.py artifacts/shadow/osBase_r1.json artifacts/shadow/osBase_r2.json -- artifacts/shadow/${c}_r1.json artifacts/shadow/${c}_r2.json | head -4; done
echo CHAINOS_DONE
