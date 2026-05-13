#!/bin/bash
if [ -n "${1:-}" ] && [[ "$1" != --* ]]; then
  TESTID="$1"
  EXTRA_ARGS=("${@:2}")
else
  TESTID="${TESTID:-95}"
  EXTRA_ARGS=("$@")
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

load_env_defaults() {
  local env_file="$1"
  [ -f "$env_file" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      ""|\#*) continue ;;
    esac
    local key="${line%%=*}"
    local value="${line#*=}"
    if [ -z "${!key+x}" ]; then
      export "$key=$value"
    fi
  done < "$env_file"
}

load_env_defaults "$ROOT/.env"

export FLOORSET_GNN_CHECKPOINT="${FLOORSET_GNN_CHECKPOINT:-$ROOT/checkpoints/gnn_best_0512_ns200000_ep10_h192_l6_acc32.pt}"
cd "$ROOT/FloorSet/iccad2026contest"
echo "Evaluation diagnostics: cost factors, top score contributors, best/worst cost cases"
uv run iccad2026_evaluate.py \
  --evaluate "$ROOT/src/architecture_v4_optimizer.py" \
  --test-id $TESTID \
  --verbose \
  "${EXTRA_ARGS[@]}"
