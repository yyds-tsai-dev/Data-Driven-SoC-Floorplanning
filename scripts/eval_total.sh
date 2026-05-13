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
    if [ "$key" = "FLOORSET_GNN_CHECKPOINT" ] || [ -z "${!key+x}" ]; then
      export "$key=$value"
    fi
  done < "$env_file"
}

load_env_defaults "$ROOT/.env"

# Usage:
#   bash scripts/eval_total.sh                         # use FLOORSET_GNN_CHECKPOINT/.env, else gnn_best_0512_ns200000_ep10_h192_l6_acc32.pt
#   bash scripts/eval_total.sh gnn_epoch10.pt          # use checkpoints/gnn_epoch10.pt
#   bash scripts/eval_total.sh checkpoints/model.pt    # use repo-relative checkpoint path
#   bash scripts/eval_total.sh /path/to/model.pt       # use absolute checkpoint path
#   bash scripts/eval_total.sh gnn_epoch10.pt --output eval.json
#   bash scripts/eval_total.sh --output eval.json      # use default checkpoint, custom output

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

if [ -n "${1:-}" ] && [[ "$1" == --* ]]; then
  if [ -z "${FLOORSET_GNN_CHECKPOINT:-}" ]; then
    export FLOORSET_GNN_CHECKPOINT="$ROOT/checkpoints/gnn_best_0512_ns200000_ep10_h192_l6_acc32.pt"
  fi
  export FLOORSET_GNN_CHECKPOINT="$(resolve_ckpt_path "$FLOORSET_GNN_CHECKPOINT")"
  EXTRA_ARGS=("$@")
elif [ -n "${1:-}" ]; then
  export FLOORSET_GNN_CHECKPOINT="$(resolve_ckpt_path "$1")"
  export FLOORSET_GNN_CHECKPOINT_SOURCE="cli"
  EXTRA_ARGS=("${@:2}")
elif [ -z "${FLOORSET_GNN_CHECKPOINT:-}" ]; then
  export FLOORSET_GNN_CHECKPOINT="$ROOT/checkpoints/gnn_best_0512_ns200000_ep10_h192_l6_acc32.pt"
  EXTRA_ARGS=()
else
  export FLOORSET_GNN_CHECKPOINT="$(resolve_ckpt_path "$FLOORSET_GNN_CHECKPOINT")"
  EXTRA_ARGS=()
fi
export FLOORSET_GNN_CHECKPOINT_SOURCE="${FLOORSET_GNN_CHECKPOINT_SOURCE:-dotenv}"

echo "Using checkpoint: $FLOORSET_GNN_CHECKPOINT"
echo "Evaluation diagnostics: cost factors, top score contributors, best/worst cost cases"

cd "$ROOT/FloorSet/iccad2026contest"

uv run iccad2026_evaluate.py \
  --evaluate "$ROOT/src/architecture_v4_optimizer.py" \
  --verbose \
  "${EXTRA_ARGS[@]}"
