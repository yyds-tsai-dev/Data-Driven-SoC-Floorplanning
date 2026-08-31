#!/bin/bash
# Final confirmation chain (Sec.17y rule: full candidate env vs current package env, one chain, x4, rotated order).
# Arms: A=base(pack env, ft0828 300k EMA, mid table)  B=FT2 (round-2 T12 EMA)  C=k20 (tail budget table)  D=FT2+k20.
# Waits for round-2 step 250k, exports EMA-only, preflights, then 4 reps x 4 arms x 4 suites (run_gate5: off,v3,v5,v6).
cd /ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning
CKD=artifacts/flow_ft_0829/flow_ft0829_ft300k_tailT12_lr1e-5_wu1k_bs12_s250k
until [ -f $CKD/step_00250000.pt ]; do sleep 60; done
sleep 120
EXP=$PWD/artifacts/flow_ft_0829/flow_ft0829_tailT12_lr1e-5_250k_ema.pt
uv run python scratchpad/flow_ft_0828/export_ema_only.py $CKD/step_00250000.pt $EXP || { echo EXPORT_FAIL; exit 1; }
PYTHONPATH=$PWD/partner:$PWD/FloorSet/iccad2026contest:$PWD/FloorSet uv run python - "$EXP" <<'PY' || { echo PREFLIGHT_FAIL; exit 1; }
import sys, torch
from flow_matching_train import checkpoint_method
from direct_diffusion_model import DirectDenoiser, DirectModelConfig
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
m = checkpoint_method(ck); assert int(ck.get("step", 0)) == 250000, ck.get("step")
cfg = DirectModelConfig(**{k: v for k, v in ck["model_config"].items() if k in DirectModelConfig.__dataclass_fields__})
model = DirectDenoiser(cfg); model.load_state_dict(ck.get("ema") or ck["model"])
print(f"PREFLIGHT_OK method={m} step={ck.get('step')} params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
PY
while pgrep -f "iccad2026_evaluate.py" >/dev/null; do sleep 20; done
export CUDA_VISIBLE_DEVICES=3
G=scripts/gate/run_gate5.sh
TABM=$(cat artifacts/p0_newbox/budget_table_mid.txt)
T20=$(cat scratchpad/rtaware/budget_table_k20.txt)
FT1=$PWD/submission/cadc1013/checkpoints/flow_matching_ft0828_tailT24_300k_ema.pt
CK=(DIRECT_OFF= "DIRECT_CKPT=$PWD/artifacts/icdc_topology/checkpoints_s2_20k/best.pt" PARTNER_DIRECT_SEAT_FIX=1 PARTNER_NREF=9 PARTNER_FLOW_SLOTS=10 PARTNER_REFINE_RES_FRAC=0.45 PARTNER_WALL_REPAIR=1 PARTNER_FLOW_WARM=1 PARTNER_EARLY_EXIT=1 PARTNER_REFINE_SECURE_FALLBACK=1 PARTNER_SEAT_R0_ADAPT=0 PARTNER_COORD_POLISH= PARTNER_FLOW_ANTITHETIC=1)
arm(){ case $1 in
  A) bash $G fnA_r$2 "${CK[@]}" "FLOW_CKPT=$FT1" "PARTNER_BUDGET_TABLE=$TABM";;
  B) bash $G fnB_r$2 "${CK[@]}" "FLOW_CKPT=$EXP" "PARTNER_BUDGET_TABLE=$TABM";;
  C) bash $G fnC_r$2 "${CK[@]}" "FLOW_CKPT=$FT1" "PARTNER_BUDGET_TABLE=$T20";;
  D) bash $G fnD_r$2 "${CK[@]}" "FLOW_CKPT=$EXP" "PARTNER_BUDGET_TABLE=$T20";;
esac; }
for a in A B C D; do arm $a 1; done; echo FN_REP1_DONE
for a in D C B A; do arm $a 2; done; echo FN_REP2_DONE
for a in B D A C; do arm $a 3; done; echo FN_REP3_DONE
for a in C A D B; do arm $a 4; done; echo FN_REP4_DONE
for s in off v3 v5 v6; do for c in B C D; do echo "== $s: $c vs A"; uv run python scripts/gate/analyze_pairs.py artifacts/shadow/fnA_r1_$s.json artifacts/shadow/fnA_r2_$s.json artifacts/shadow/fnA_r3_$s.json artifacts/shadow/fnA_r4_$s.json -- artifacts/shadow/fn${c}_r1_$s.json artifacts/shadow/fn${c}_r2_$s.json artifacts/shadow/fn${c}_r3_$s.json artifacts/shadow/fn${c}_r4_$s.json | head -8; done; done
echo CHAINFINAL_DONE
