#!/bin/bash
# Pin-frame portfolio seat re-gate under the CURRENT B' env (FT2+s16): the only mechanism aimed at the locked-wall class.
# A = B' env; P = + PARTNER_PIN_FRAME_SLOTS=2 PARTNER_PIN_FRAME_SLOT_MAX_UTIL=0.75 (Sec.17d best arm). 4 suites x 4 reps alternating.
cd /ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning
while pgrep -f "iccad2026_evaluate.py" >/dev/null; do sleep 20; done
export CUDA_VISIBLE_DEVICES=3
G=scripts/gate/run_gate5.sh
FT2=$PWD/artifacts/flow_ft_0829/flow_ft0829_tailT12_lr1e-5_250k_ema.pt
TABM=$(cat artifacts/p0_newbox/budget_table_mid.txt)
CK=(DIRECT_OFF= "DIRECT_CKPT=$PWD/artifacts/icdc_topology/checkpoints_s2_20k/best.pt" PARTNER_DIRECT_SEAT_FIX=1 PARTNER_REFINE_RES_FRAC=0.45 PARTNER_WALL_REPAIR=1 PARTNER_FLOW_WARM=1 PARTNER_EARLY_EXIT=1 PARTNER_REFINE_SECURE_FALLBACK=1 PARTNER_SEAT_R0_ADAPT=0 PARTNER_COORD_POLISH= PARTNER_FLOW_ANTITHETIC=1 "FLOW_CKPT=$FT2" "PARTNER_BUDGET_TABLE=$TABM" PARTNER_NREF=12 PARTNER_FLOW_SLOTS=16)
A(){ bash $G pfA_r$1 "${CK[@]}"; }
P(){ bash $G pfP_r$1 "${CK[@]}" PARTNER_PIN_FRAME_SLOTS=2 PARTNER_PIN_FRAME_SLOT_MAX_UTIL=0.75; }
A 1; P 1; P 2; A 2; A 3; P 3; P 4; A 4
for s in off v3 v5 v6; do echo "== $s PF vs base"; uv run python scripts/gate/analyze_pairs.py artifacts/shadow/pfA_r{1,2,3,4}_$s.json -- artifacts/shadow/pfP_r{1,2,3,4}_$s.json | head -4; done
echo CHAINPF_DONE
