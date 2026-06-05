#!/bin/bash
if [ -n "${1:-}" ] && [[ "$1" != --* ]]; then
  TESTID="$1"
  EXTRA_ARGS=("${@:2}")
else
  TESTID="${TESTID:-95}"
  EXTRA_ARGS=("$@")
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EVALUATOR="$ROOT/scripts/iccad2026_evaluate.py"
OPTIMIZER="$ROOT/src/architecture_v5_optimizer.py"
DEFAULT_CKPT="$ROOT/checkpoints/gnn_transformer_best_0521_ns1000000_ep3_encgraph_transformer_h256_l6_acc32_heads8.pt"

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

export FLOORSET_GNN_CHECKPOINT="${FLOORSET_GNN_CHECKPOINT:-$DEFAULT_CKPT}"
export FLOORSET_GNN_CHECKPOINT="$(resolve_ckpt_path "$FLOORSET_GNN_CHECKPOINT")"
export FLOORSET_GNN_CHECKPOINT_SOURCE="${FLOORSET_GNN_CHECKPOINT_SOURCE:-dotenv}"
cd "$ROOT/FloorSet/iccad2026contest"
echo "Using checkpoint: $FLOORSET_GNN_CHECKPOINT"
echo "Using evaluator: $EVALUATOR"
echo "Evaluation diagnostics: cost factors, top score contributors, best/worst cost cases"
export PYTHONPATH="$ROOT/FloorSet/iccad2026contest:$ROOT/FloorSet:${PYTHONPATH:-}"
uv run "$EVALUATOR" \
  --data-path ../ \
  --evaluate "$OPTIMIZER" \
  --test-id "$TESTID" \
  --verbose \
  "${EXTRA_ARGS[@]}"
