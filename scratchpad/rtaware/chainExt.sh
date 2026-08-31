#!/bin/bash
# B' env (FT2 + s16, mid table) on the competitor-provided panels (selfgen 1x/2x/3x, proxy x6) + public/v3/v5/v6. One rep each.
cd /ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning
while pgrep -f "iccad2026_evaluate.py" >/dev/null; do sleep 20; done
export CUDA_VISIBLE_DEVICES=3
S=scripts/gate/run_shadow.sh
FT2=$PWD/artifacts/flow_ft_0829/flow_ft0829_tailT12_lr1e-5_250k_ema.pt
TABM=$(cat artifacts/p0_newbox/budget_table_mid.txt)
CK=(DIRECT_OFF= "DIRECT_CKPT=$PWD/artifacts/icdc_topology/checkpoints_s2_20k/best.pt" PARTNER_DIRECT_SEAT_FIX=1 PARTNER_REFINE_RES_FRAC=0.45 PARTNER_WALL_REPAIR=1 PARTNER_FLOW_WARM=1 PARTNER_EARLY_EXIT=1 PARTNER_REFINE_SECURE_FALLBACK=1 PARTNER_SEAT_R0_ADAPT=0 PARTNER_COORD_POLISH= PARTNER_FLOW_ANTITHETIC=1 "FLOW_CKPT=$FT2" "PARTNER_BUDGET_TABLE=$TABM" PARTNER_NREF=12 PARTNER_FLOW_SLOTS=16)
SG=$PWD/shadow_hidden/ext_selfgen/self_gen_v2_share; PX=$PWD/shadow_hidden/ext_proxy/proxy
bash $S extB_sg1x $SG/1x "${CK[@]}"
bash $S extB_sg2x $SG/2x "${CK[@]}"
bash $S extB_sg3x $SG/3x "${CK[@]}"
for g in bbox50 blockarea50 blockar50; do for c in p2b50_b2b3p7_beta3p123 p2b0p5_b2b0p5_beta3p123; do
  short=$(echo $c | sed 's/p2b50_b2b3p7_beta3p123/short/;s/p2b0p5_b2b0p5_beta3p123/flat/')
  bash $S extB_px_${g}_${short} $PX/$g/$c "${CK[@]}"
done; done
bash $S extB_off $PWD/FloorSet "${CK[@]}"
bash $S extB_v3 $PWD/shadow_hidden/shadow_hidden_v3_share "${CK[@]}"
bash $S extB_v5 $PWD/shadow_hidden/shadow_hidden_v5_share "${CK[@]}"
bash $S extB_v6 $PWD/shadow_hidden/shadow_hidden_v6_share "${CK[@]}"
echo CHAINEXT_DONE
