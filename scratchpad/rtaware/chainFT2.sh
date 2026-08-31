#!/bin/bash
# FT round-2 gate: wait for step_00250000.pt, export EMA-only, preflight, then
# 4 suites x (pack model vs ft2) x2 reps, arm order flipped in rep 2. Package env (mid table).
cd /ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning
CKD=artifacts/flow_ft_0829/flow_ft0829_ft300k_tailT12_lr1e-5_wu1k_bs12_s250k
until [ -f $CKD/step_00250000.pt ]; do sleep 60; done
sleep 90   # let the writer finish
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
FT1=$PWD/submission/cadc1013/checkpoints/flow_matching_ft0828_tailT24_300k_ema.pt
CK=(DIRECT_OFF= "DIRECT_CKPT=$PWD/artifacts/icdc_topology/checkpoints_s2_20k/best.pt" PARTNER_DIRECT_SEAT_FIX=1 PARTNER_NREF=9 PARTNER_FLOW_SLOTS=10 PARTNER_REFINE_RES_FRAC=0.45 PARTNER_WALL_REPAIR=1 PARTNER_FLOW_WARM=1 PARTNER_EARLY_EXIT=1 PARTNER_REFINE_SECURE_FALLBACK=1 PARTNER_SEAT_R0_ADAPT=0 PARTNER_COORD_POLISH= PARTNER_FLOW_ANTITHETIC=1 "PARTNER_BUDGET_TABLE=$TABM")
bash $G ft2Base_r1 "${CK[@]}" "FLOW_CKPT=$FT1"
bash $G ft2T12_r1 "${CK[@]}" "FLOW_CKPT=$EXP"
bash $G ft2T12_r2 "${CK[@]}" "FLOW_CKPT=$EXP"
bash $G ft2Base_r2 "${CK[@]}" "FLOW_CKPT=$FT1"
for s in off v3 v5 v6; do echo "== $s"; uv run python scripts/gate/analyze_pairs.py artifacts/shadow/ft2Base_r1_$s.json artifacts/shadow/ft2Base_r2_$s.json -- artifacts/shadow/ft2T12_r1_$s.json artifacts/shadow/ft2T12_r2_$s.json | head -12; done
echo CHAINFT2_DONE
