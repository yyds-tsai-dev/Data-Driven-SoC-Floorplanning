#!/bin/bash
TESTID="${1:-95}"
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

export FLOORSET_GNN_CHECKPOINT="${FLOORSET_GNN_CHECKPOINT:-$ROOT/checkpoints/gnn_best.pt}"
cd "$ROOT/FloorSet/iccad2026contest"
uv run iccad2026_evaluate.py \
  --evaluate "$ROOT/src/architecture_v4_optimizer.py" \
  --test-id $TESTID \
  --verbose
