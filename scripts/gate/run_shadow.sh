#!/bin/bash
# Usage: bash run_shadow.sh <tag> <data_root_containing_LiteTensorDataTest> [KEY=VAL ...]
set -e
ROOT=/ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning
TAG="$1"; DATA="$2"; shift 2
export PYTHONPATH="$ROOT/FloorSet/iccad2026contest:$ROOT/FloorSet" MPLCONFIGDIR=/tmp/floorset-shadow-mpl
export VKILL_OFF=1 PARTNER_POOL=24 PARTNER_NREF=6 PARTNER_NREF_MIN_N=95 PARTNER_DIRECT_MIN=0.3 PARTNER_DIRECT_SEAT_FIX=0 PARTNER_OVERSAMPLE=1 PARTNER_KS_CAP=6 PARTNER_DIRECT_SOLVER=dpmpp PARTNER_DDIM_STEPS=2 PARTNER_FLOW_SOLVER=euler PARTNER_FLOW_STEPS=8 PARTNER_FLOW_ANTITHETIC=1 PARTNER_PRESCREEN_V=1 PARTNER_TAG_ANCHOR_EXTRA=3 PARTNER_BUDGET_SCALE=8.498e-5 PARTNER_BUDGET_TAU=12 PARTNER_BUDGET_MIN=0.05 PARTNER_BUDGET_MAX=1.22 PARTNER_POOL_GATE=0 PARTNER_REFINE_STALL_STOP=1 PARTNER_SA_KERNEL=numba PARTNER_REFINE_KERNEL=numba PARTNER_REFINE_FASTBUILD=1 PARTNER_FAST_SETUP=1 PARTNER_EDGE_SEAT_V2=1 PARTNER_FRAME_WPIN=1 PARTNER_COORD_POLISH=1 PARTNER_FRAME_SCALE_LADDER=1 PARTNER_FRAME_SCALE_SET=1.02 PARTNER_SEAT_FINAL=1 PARTNER_TAG_COMPRESS=1 PARTNER_GROUP_BRIDGE=1 PARTNER_GPU_ARM=0 PARTNER_RETRIEVAL_SLOTS=0 PYTHONHASHSEED=0
for kv in "$@"; do export "$kv"; done
OUT="$ROOT/artifacts/shadow"; mkdir -p "$OUT"
echo "TAG=$TAG DATA=$DATA"; echo "load_before: $(uptime)"
cd "$ROOT/FloorSet/iccad2026contest"
uv run python "$ROOT/scripts/iccad2026_evaluate.py" --data-path "$DATA" --evaluate "$ROOT/partner/contest_optimizer.py" --output "$OUT/${TAG}.json" 2>&1 | tee "$OUT/${TAG}.log" | tail -4
grep -h "\[selfcheck\]\|\[flowtail\]\|\[pool-fallback\]" "$OUT/${TAG}.log" | head -3
uv run python - "$OUT/${TAG}.json" <<'PY'
import json, sys, math
d = json.load(open(sys.argv[1])); s = d.get("summary", {})
rs = d["test_results"]; mx = max(r["block_count"] for r in rs); W = sum(math.exp((r["block_count"]-mx)/12) for r in rs)
w = lambda k: sum(math.exp((r["block_count"]-mx)/12)*r.get(k,0) for r in rs)/W
print("RESULT tag=%s noRT=%.4f feasible=%s avg_rt=%.3f p90=%.3f max=%.3f | hpwl=%.4f area=%.4f v_rel=%.4f bnd=%.2f grp=%.2f mib=%.2f" % (
  sys.argv[1].rsplit("/",1)[-1], d["total_score_no_runtime"], s.get("num_feasible"), s.get("avg_runtime",0), s.get("p90_runtime",0), s.get("max_runtime",0),
  w("hpwl_gap"), w("area_gap"), w("violations_relative"), w("boundary_violations"), w("grouping_violations"), w("mib_violations")))
PY
