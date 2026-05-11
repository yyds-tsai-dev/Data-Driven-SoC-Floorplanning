#!/bin/bash

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

# Usage:
#   ./script.sh                 # default: gnn_best.pt
#   ./script.sh gnn_epoch10.pt
#   ./script.sh /path/to/model.pt

CKPT_NAME="${1:-gnn_best.pt}"

if [[ "$CKPT_NAME" = /* ]]; then
  CKPT_PATH="$CKPT_NAME"
else
  CKPT_PATH="$ROOT/checkpoints/$CKPT_NAME"
fi

if [ -z "${FLOORSET_GNN_CHECKPOINT:-}" ] || [ "${FLOORSET_GNN_CHECKPOINT:-}" = "checkpoints/gnn_best.pt" ]; then
  export FLOORSET_GNN_CHECKPOINT="$CKPT_PATH"
fi

cd "$ROOT/FloorSet/iccad2026contest"

uv run iccad2026_evaluate.py \
  --evaluate "$ROOT/src/architecture_v4_optimizer.py" \
  --verbose
