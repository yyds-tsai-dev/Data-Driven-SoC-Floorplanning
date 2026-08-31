#!/bin/bash
# Pool seeding width: s16 (slots16/nref12, current B') vs slots 24 / nref 12 (all 24 pool restarts Flow-seeded). Official x2 paired reversed, FT2, mid table.
cd /ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning
while pgrep -f "iccad2026_evaluate.py" >/dev/null; do sleep 20; done
export CUDA_VISIBLE_DEVICES=3
S=scripts/gate/run_shadow.sh; OFF=$PWD/FloorSet
FT2=$PWD/artifacts/flow_ft_0829/flow_ft0829_tailT12_lr1e-5_250k_ema.pt
TABM=$(cat artifacts/p0_newbox/budget_table_mid.txt)
CK=(DIRECT_OFF= "DIRECT_CKPT=$PWD/artifacts/icdc_topology/checkpoints_s2_20k/best.pt" PARTNER_DIRECT_SEAT_FIX=1 PARTNER_REFINE_RES_FRAC=0.45 PARTNER_WALL_REPAIR=1 PARTNER_FLOW_WARM=1 PARTNER_EARLY_EXIT=1 PARTNER_REFINE_SECURE_FALLBACK=1 PARTNER_SEAT_R0_ADAPT=0 PARTNER_COORD_POLISH= PARTNER_FLOW_ANTITHETIC=1 "FLOW_CKPT=$FT2" "PARTNER_BUDGET_TABLE=$TABM" PARTNER_NREF=12)
bash $S s24Base_r1 $OFF "${CK[@]}" PARTNER_FLOW_SLOTS=16
bash $S s24W_r1 $OFF "${CK[@]}" PARTNER_FLOW_SLOTS=24
bash $S s24W_r2 $OFF "${CK[@]}" PARTNER_FLOW_SLOTS=24
bash $S s24Base_r2 $OFF "${CK[@]}" PARTNER_FLOW_SLOTS=16
uv run python scripts/gate/analyze_pairs.py artifacts/shadow/s24Base_r1.json artifacts/shadow/s24Base_r2.json -- artifacts/shadow/s24W_r1.json artifacts/shadow/s24W_r2.json | head -6
echo CHAINS24_DONE
