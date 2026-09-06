#!/bin/bash
# Three-suite gate: official100 + shadow v3 + shadow alpha_1 with one env config.
# Usage: bash run_gate3.sh <tag> [KEY=VAL ...]   -> prints 3 RESULT lines + GATE3 mean
set -e
ROOT=/ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning
TAG="$1"; shift
S=$ROOT/scripts/gate/run_shadow.sh
bash $S "${TAG}_off" "$ROOT/FloorSet" "$@"
bash $S "${TAG}_v3" "$ROOT/artifacts/shadow_hidden_suites/shadow_hidden_v3_share" "$@"
bash $S "${TAG}_a1" "$ROOT/artifacts/shadow_hidden_suites/alpha_1/alpha_1" "$@"
cd "$ROOT" && uv run python - "$TAG" <<'PY'
import json, sys
t=sys.argv[1]; v=[json.load(open(f"artifacts/shadow_gate_runs/{t}_{s}.json"))["total_score_no_runtime"] for s in ("off","v3","a1")]
print("GATE3 tag=%s off=%.4f v3=%.4f a1=%.4f mean=%.4f" % (t, *v, sum(v)/3))
PY
