#!/bin/bash
# Ladder census under the SHIPPING package env, 300k ckpt vs v1, GPU 0.
cd /ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning
T=/ldaphome/yyds-tsai-dev/.claude/jobs/06af0e53/tmp/seatfix
R=$PWD
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$R/FloorSet/iccad2026contest:$R/FloorSet" MPLCONFIGDIR=/tmp/floorset-seatfix-mpl
export VKILL_OFF=1 PARTNER_POOL=24 PARTNER_NREF_MIN_N=95 PARTNER_DIRECT_MIN=0.3 PARTNER_OVERSAMPLE=1 PARTNER_KS_CAP=6 PARTNER_DIRECT_SOLVER=dpmpp PARTNER_DDIM_STEPS=2 PARTNER_FLOW_SOLVER=euler PARTNER_FLOW_STEPS=8 PARTNER_FLOW_ANTITHETIC=1 PARTNER_PRESCREEN_V=1 PARTNER_TAG_ANCHOR_EXTRA=3 PARTNER_BUDGET_SCALE=8.498e-5 PARTNER_BUDGET_TAU=12 PARTNER_BUDGET_MIN=0.05 PARTNER_BUDGET_MAX=1.22 PARTNER_POOL_GATE=0 PARTNER_REFINE_STALL_STOP=1 PARTNER_SA_KERNEL=numba PARTNER_REFINE_KERNEL=numba PARTNER_REFINE_FASTBUILD=1 PARTNER_FAST_SETUP=1 PARTNER_EDGE_SEAT_V2=1 PARTNER_FRAME_WPIN=1 PARTNER_FRAME_SCALE_LADDER=1 PARTNER_FRAME_SCALE_SET=1.02 PARTNER_SEAT_FINAL=1 PARTNER_TAG_COMPRESS=1 PARTNER_GROUP_BRIDGE=1 PARTNER_GPU_ARM=0 PARTNER_RETRIEVAL_SLOTS=0 PYTHONHASHSEED=0
export "PARTNER_BUDGET_TABLE=$(cat artifacts/p0_newbox/budget_table_mid.txt)" DIRECT_OFF= "DIRECT_CKPT=$R/artifacts/icdc_topology/checkpoints_s2_20k/best.pt"
export PARTNER_DIRECT_SEAT_FIX=1 PARTNER_NREF=9 PARTNER_FLOW_SLOTS=10 PARTNER_REFINE_RES_FRAC=0.45 PARTNER_WALL_REPAIR=1 PARTNER_FLOW_WARM=1 PARTNER_EARLY_EXIT=1 PARTNER_REFINE_SECURE_FALLBACK=1 PARTNER_SEAT_R0_ADAPT=0
export PARTNER_COORD_POLISH=
export PARTNER_LADDER_DEBUG=1 PARTNER_SECURE_FALLBACK_DEBUG=1 PARTNER_SEAT_DEBUG=1
run() {  # $1 tag, $2 ckpt, rest = extra env
  local tag=$1; local ck=$2; shift 2
  for kv in "$@"; do export "$kv"; done
  export FLOW_CKPT=$ck; ( cd FloorSet/iccad2026contest && uv run python $R/scripts/iccad2026_evaluate.py --data-path $R/shadow_hidden/shadow_hidden_v3_share --evaluate $R/partner/contest_optimizer.py --output $T/${tag}.json ) > $T/${tag}.out 2> $T/${tag}.err
  uv run --no-project python - $T/${tag}.json $tag <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); s=d["summary"]
print("CENSUS tag=%s noRT=%.4f feas=%s avg=%.3f p90=%.3f max=%.3f"%(sys.argv[2],d["total_score_no_runtime"],s["num_feasible"],s["avg_runtime"],s["p90_runtime"],s["max_runtime"]))
PY
}
FT=$R/submission/cadc1013/checkpoints/flow_matching_ft0828_tailT24_300k_ema.pt
V1=$R/submission/cadc1013/checkpoints/flow_matching_v1_final.pt
run cen300v3_r1 $FT

echo SEATFIX_CENSUS_DONE
