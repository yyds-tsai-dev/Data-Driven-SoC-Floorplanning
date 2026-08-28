#!/bin/bash
# Flow fine-tune gate: v1 vs newest fine-tune snapshot, five suites, x2 paired, arm order alternated.
cd /ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning
T=/ldaphome/yyds-tsai-dev/.claude/jobs/06af0e53/tmp
until grep -q CHAINN_DONE $T/chainN.log 2>/dev/null; do sleep 30; done
while pgrep -f "iccad2026_evaluate.py" >/dev/null; do sleep 20; done
CKD=artifacts/flow_ft_0828/flow_ft0828_ftv1_tailT24_w0-89_lr2e-5_wu1k_bs12_s300k
SNAP=$(ls $CKD/step_*.pt 2>/dev/null | sort | tail -1)
[ -n "$SNAP" ] || { echo "NO_SNAPSHOT"; exit 1; }
STEP=$(basename $SNAP .pt | sed 's/step_0*//'); STEPK=$((STEP/1000))
EXP=$PWD/artifacts/flow_ft_0828/flow_ft0828_tailT24_lr2e-5_step${STEPK}k.pt
echo "SNAP=$SNAP STEPK=$STEPK EXP=$EXP"
uv run python artifacts/flow_ft_0828/export_gate_ckpt.py $SNAP $EXP || { echo EXPORT_FAIL; exit 1; }
# preflight: load exactly as contest_optimizer._load_flow_model does (its failure is silent)
PYTHONPATH=$PWD/partner:$PWD/FloorSet/iccad2026contest:$PWD/FloorSet uv run python - "$EXP" <<'PY' || { echo PREFLIGHT_FAIL; exit 1; }
import sys, torch
from flow_matching_train import checkpoint_method
from direct_diffusion_model import DirectDenoiser, DirectModelConfig
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
m = checkpoint_method(ck)
cfg = DirectModelConfig(**{k: v for k, v in ck["model_config"].items() if k in DirectModelConfig.__dataclass_fields__})
model = DirectDenoiser(cfg); model.load_state_dict(ck.get("ema") or ck["model"])
print(f"PREFLIGHT_OK method={m} step={ck.get('step')} params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
PY
G=scripts/gate/run_gate5.sh
TABM=$(cat artifacts/p0_newbox/budget_table_mid.txt)
export CUDA_VISIBLE_DEVICES=3
CK=(DIRECT_OFF= "DIRECT_CKPT=$PWD/artifacts/icdc_topology/checkpoints_s2_20k/best.pt" PARTNER_DIRECT_SEAT_FIX=1 PARTNER_NREF=9 PARTNER_FLOW_SLOTS=10 PARTNER_REFINE_RES_FRAC=0.45 PARTNER_WALL_REPAIR=1 PARTNER_FLOW_WARM=1 PARTNER_EARLY_EXIT=1 PARTNER_REFINE_SECURE_FALLBACK=1 PARTNER_SEAT_R0_ADAPT=0 PARTNER_COORD_POLISH= "PARTNER_BUDGET_TABLE=$TABM")
V1=$PWD/submission/cadc1013/checkpoints/flow_matching_v1_final.pt
bash $G ftBase_r1 "${CK[@]}" "FLOW_CKPT=$V1"
bash $G ft${STEPK}k_r1 "${CK[@]}" "FLOW_CKPT=$EXP"
bash $G ft${STEPK}k_r2 "${CK[@]}" "FLOW_CKPT=$EXP"
bash $G ftBase_r2 "${CK[@]}" "FLOW_CKPT=$V1"
uv run python scripts/gate/analyze_pairs.py artifacts/shadow/ftBase_r1_off.json artifacts/shadow/ftBase_r2_off.json -- artifacts/shadow/ft${STEPK}k_r1_off.json artifacts/shadow/ft${STEPK}k_r2_off.json | head -30
echo CHAINFT_DONE
