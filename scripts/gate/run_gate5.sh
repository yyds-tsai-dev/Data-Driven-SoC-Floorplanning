#!/bin/bash
# Five-suite gate: official100 + shadow v3 + v5 + v6 + alpha_1 with one env config.
# Usage: bash run_gate5.sh <tag> [KEY=VAL ...]
set -e
ROOT=/ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning
TAG="$1"; shift
S=$ROOT/scripts/gate/run_shadow.sh
bash $S "${TAG}_off" "$ROOT/FloorSet" "$@"
bash $S "${TAG}_v3" "$ROOT/shadow_hidden/shadow_hidden_v3_share" "$@"
bash $S "${TAG}_v5" "$ROOT/shadow_hidden/shadow_hidden_v5_share" "$@"
bash $S "${TAG}_v6" "$ROOT/shadow_hidden/shadow_hidden_v6_share" "$@"
bash $S "${TAG}_a1" "$ROOT/shadow_hidden/alpha_1/alpha_1" "$@"
cd "$ROOT" && uv run python - "$TAG" <<'PY'
import json, sys
t=sys.argv[1]; v=[json.load(open(f"artifacts/shadow/{t}_{s}.json"))["total_score_no_runtime"] for s in ("off","v3","v5","v6","a1")]
print("GATE5 tag=%s off=%.4f v3=%.4f v5=%.4f v6=%.4f a1=%.4f mean4=%.4f" % (t, *v, sum(v[:4])/4))
PY
