#!/bin/bash
# Interaction gate: under the k20 tail table, does FLOW_SLOTS=16/NREF=12 still help? A = FT2+k20 (slots 10/nref 9), W = FT2+k20+slots16. 4 suites x 4 reps alternating.
cd /ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning
T=/ldaphome/yyds-tsai-dev/.claude/jobs/06af0e53/tmp
until grep -q CHAINW16_DONE $T/chainW16.log 2>/dev/null; do sleep 30; done
while pgrep -f "iccad2026_evaluate.py" >/dev/null; do sleep 20; done
export CUDA_VISIBLE_DEVICES=3
G=scripts/gate/run_gate5.sh
FT2=$PWD/artifacts/flow_ft_0829/flow_ft0829_tailT12_lr1e-5_250k_ema.pt
T20=$(cat scratchpad/rtaware/budget_table_k20.txt)
CK=(DIRECT_OFF= "DIRECT_CKPT=$PWD/artifacts/icdc_topology/checkpoints_s2_20k/best.pt" PARTNER_DIRECT_SEAT_FIX=1 PARTNER_REFINE_RES_FRAC=0.45 PARTNER_WALL_REPAIR=1 PARTNER_FLOW_WARM=1 PARTNER_EARLY_EXIT=1 PARTNER_REFINE_SECURE_FALLBACK=1 PARTNER_SEAT_R0_ADAPT=0 PARTNER_COORD_POLISH= PARTNER_FLOW_ANTITHETIC=1 "FLOW_CKPT=$FT2" "PARTNER_BUDGET_TABLE=$T20")
A(){ bash $G k16A_r$1 "${CK[@]}" PARTNER_NREF=9 PARTNER_FLOW_SLOTS=10; }
W(){ bash $G k16W_r$1 "${CK[@]}" PARTNER_NREF=12 PARTNER_FLOW_SLOTS=16; }
W 1; A 1; A 2; W 2; W 3; A 3; A 4; W 4
for s in off v3 v5 v6; do echo "== $s W vs A (k20 table)"; uv run python scripts/gate/analyze_pairs.py artifacts/shadow/k16A_r{1,2,3,4}_$s.json -- artifacts/shadow/k16W_r{1,2,3,4}_$s.json | head -6; done
echo CHAINK16_DONE
