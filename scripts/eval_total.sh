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
#   bash scripts/eval_total.sh                         # use FLOORSET_GNN_CHECKPOINT/.env, else gnn_best.pt
#   bash scripts/eval_total.sh gnn_epoch10.pt          # use checkpoints/gnn_epoch10.pt
#   bash scripts/eval_total.sh checkpoints/model.pt    # use repo-relative checkpoint path
#   bash scripts/eval_total.sh /path/to/model.pt       # use absolute checkpoint path

resolve_ckpt_path() {
  local ckpt="$1"
  if [[ "$ckpt" = /* ]]; then
    printf '%s\n' "$ckpt"
  elif [[ "$ckpt" = */* ]]; then
    printf '%s\n' "$ROOT/$ckpt"
  else
    printf '%s\n' "$ROOT/checkpoints/$ckpt"
  fi
}

if [ -n "${1:-}" ]; then
  export FLOORSET_GNN_CHECKPOINT="$(resolve_ckpt_path "$1")"
elif [ -z "${FLOORSET_GNN_CHECKPOINT:-}" ]; then
  export FLOORSET_GNN_CHECKPOINT="$ROOT/checkpoints/gnn_best.pt"
fi

echo "Using checkpoint: $FLOORSET_GNN_CHECKPOINT"

cd "$ROOT/FloorSet/iccad2026contest"

uv run iccad2026_evaluate.py \
  --evaluate "$ROOT/src/architecture_v4_optimizer.py" \
  --verbose
