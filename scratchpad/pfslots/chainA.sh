#!/bin/bash
# Interleaved gate: base / plain / plain+MAXUTIL / anchored+MAXUTIL, x3
ROOT=/ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning
R=/ldaphome/yyds-tsai-dev/.claude/jobs/06af0e53/tmp/run_gate3.sh
COMMON=(PARTNER_COORD_POLISH= "PARTNER_BUDGET_TABLE=$(cat $ROOT/artifacts/p0_newbox/budget_table_mid.txt)" \
  DIRECT_OFF= "DIRECT_CKPT=$ROOT/artifacts/icdc_topology/checkpoints_s2_20k/best.pt" \
  "FLOW_CKPT=$ROOT/submission/cadc1013/checkpoints/flow_matching_v1_final.pt" \
  PARTNER_DIRECT_SEAT_FIX=1 PARTNER_NREF=9 PARTNER_FLOW_SLOTS=10 PARTNER_REFINE_RES_FRAC=0.45 \
  PARTNER_WALL_REPAIR=1 PARTNER_FLOW_WARM=1 CUDA_VISIBLE_DEVICES=3)
for r in 1 2 3; do
  echo "### rep $r  $(date -u +%H:%M:%S)  load: $(uptime | sed 's/.*average//')"
  bash $R pfa_base_r$r   "${COMMON[@]}"
  bash $R pfa_plain_r$r  "${COMMON[@]}" PARTNER_PIN_FRAME_SLOTS=2 PARTNER_PIN_FRAME_SLOT_FROM=plain
  bash $R pfa_plainu_r$r "${COMMON[@]}" PARTNER_PIN_FRAME_SLOTS=2 PARTNER_PIN_FRAME_SLOT_FROM=plain PARTNER_PIN_FRAME_SLOT_MAX_UTIL=0.75
  bash $R pfa_anchu_r$r  "${COMMON[@]}" PARTNER_PIN_FRAME_SLOTS=2 PARTNER_PIN_FRAME_SLOT_MAX_UTIL=0.75
done
echo "### DONE $(date -u +%H:%M:%S)"
