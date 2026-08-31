#!/bin/bash
# Round-3 gate: wait for step 150k, export EMA-only, preflight, then B' env (FT2+s16) vs FT3+s16, 4 suites x 4 reps alternating.
cd /ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning
CKD=artifacts/flow_ft_0830/flow_ft0830_ft250k_tailT12_lr5e-6_x02_hp06_s150k
until [ -f $CKD/step_00150000.pt ]; do sleep 60; done
sleep 120
EXP=$PWD/artifacts/flow_ft_0830/flow_ft0830_tailT12_lr5e-6_x02_hp06_150k_ema.pt
uv run python scratchpad/flow_ft_0828/export_ema_only.py $CKD/step_00150000.pt $EXP || { echo EXPORT_FAIL; exit 1; }
PYTHONPATH=$PWD/partner:$PWD/FloorSet/iccad2026contest:$PWD/FloorSet uv run python - "$EXP" <<'PY' || { echo PREFLIGHT_FAIL; exit 1; }
import sys, torch
from flow_matching_train import checkpoint_method
from direct_diffusion_model import DirectDenoiser, DirectModelConfig
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
m = checkpoint_method(ck); assert int(ck.get("step", 0)) == 150000, ck.get("step")
cfg = DirectModelConfig(**{k: v for k, v in ck["model_config"].items() if k in DirectModelConfig.__dataclass_fields__})
model = DirectDenoiser(cfg); model.load_state_dict(ck.get("ema") or ck["model"])
print(f"PREFLIGHT_OK method={m} step={ck.get('step')} params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
PY
while pgrep -f "iccad2026_evaluate.py" >/dev/null; do sleep 20; done
export CUDA_VISIBLE_DEVICES=3
G=scripts/gate/run_gate5.sh
FT2=$PWD/artifacts/flow_ft_0829/flow_ft0829_tailT12_lr1e-5_250k_ema.pt
TABM=$(cat artifacts/p0_newbox/budget_table_mid.txt)
CK=(DIRECT_OFF= "DIRECT_CKPT=$PWD/artifacts/icdc_topology/checkpoints_s2_20k/best.pt" PARTNER_DIRECT_SEAT_FIX=1 PARTNER_REFINE_RES_FRAC=0.45 PARTNER_WALL_REPAIR=1 PARTNER_FLOW_WARM=1 PARTNER_EARLY_EXIT=1 PARTNER_REFINE_SECURE_FALLBACK=1 PARTNER_SEAT_R0_ADAPT=0 PARTNER_COORD_POLISH= PARTNER_FLOW_ANTITHETIC=1 "PARTNER_BUDGET_TABLE=$TABM" PARTNER_NREF=12 PARTNER_FLOW_SLOTS=16)
A(){ bash $G ft3A_r$1 "${CK[@]}" "FLOW_CKPT=$FT2"; }
B(){ bash $G ft3B_r$1 "${CK[@]}" "FLOW_CKPT=$EXP"; }
A 1; B 1; B 2; A 2; A 3; B 3; B 4; A 4
for s in off v3 v5 v6; do echo "== $s FT3 vs FT2 (both s16)"; uv run python scripts/gate/analyze_pairs.py artifacts/shadow/ft3A_r{1,2,3,4}_$s.json -- artifacts/shadow/ft3B_r{1,2,3,4}_$s.json | head -6; done
for s in off v3 v5 v6; do printf "%s 76-89 col A->B: " $s; uv run python scripts/gate/band_pairs.py artifacts/shadow/ft3A_r{1,2,3,4}_$s.json -- artifacts/shadow/ft3B_r{1,2,3,4}_$s.json | grep -E "^ *76-89"; done
echo CHAINFT3_DONE
