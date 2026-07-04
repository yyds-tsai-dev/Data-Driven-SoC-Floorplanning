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
OPTIMIZER="$ROOT/src/architecture_v11_optimizer.py"
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

diffusion_state_label() {
  local use_ema="${FLOORSET_DIFFUSION_USE_EMA:-1}"
  case "${use_ema,,}" in
    0|false|off|no|raw) printf '%s\n' "raw" ;;
    *) printf '%s\n' "ema-preferred" ;;
  esac
}

parse_eval_args() {
  EXTRA_ARGS=()
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --diffusion-checkpoint)
        shift
        if [ -z "${1:-}" ]; then
          echo "--diffusion-checkpoint requires a path" >&2
          return 1
        fi
        export FLOORSET_DIFFUSION_CHECKPOINT="$(resolve_ckpt_path "$1")"
        export FLOORSET_DIFFUSION_CHECKPOINT_SOURCE="cli"
        ;;
      --diffusion-use-ema)
        export FLOORSET_DIFFUSION_USE_EMA="1"
        ;;
      --diffusion-use-raw)
        export FLOORSET_DIFFUSION_USE_EMA="0"
        ;;
      --gnn-checkpoint)
        shift
        if [ -z "${1:-}" ]; then
          echo "--gnn-checkpoint requires a path" >&2
          return 1
        fi
        export FLOORSET_GNN_CHECKPOINT="$(resolve_ckpt_path "$1")"
        export FLOORSET_GNN_CHECKPOINT_SOURCE="cli"
        ;;
      *)
        EXTRA_ARGS+=("$1")
        ;;
    esac
    shift
  done
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
parse_eval_args "${EXTRA_ARGS[@]}" || exit $?

export FLOORSET_GNN_CHECKPOINT="${FLOORSET_GNN_CHECKPOINT:-$DEFAULT_CKPT}"
export FLOORSET_GNN_CHECKPOINT="$(resolve_ckpt_path "$FLOORSET_GNN_CHECKPOINT")"
export FLOORSET_GNN_CHECKPOINT_SOURCE="${FLOORSET_GNN_CHECKPOINT_SOURCE:-dotenv}"
FLOORPLAN_DIR="${FLOORSET_EVAL_FLOORPLAN_DIR:-$ROOT/artifacts/eval_v11/floorplans/single_case_${TESTID}}"
cd "$ROOT/FloorSet/iccad2026contest"
echo "Using GNN fallback checkpoint: $FLOORSET_GNN_CHECKPOINT"
echo "Using diffusion checkpoint: ${FLOORSET_DIFFUSION_CHECKPOINT:-<none>}"
echo "Using diffusion checkpoint state: $(diffusion_state_label)"
echo "Using evaluator: $EVALUATOR"
echo "Floorplan PNG output: $FLOORPLAN_DIR"
echo "Evaluation diagnostics: cost factors, top score contributors, best/worst cost cases"
export PYTHONPATH="$ROOT/FloorSet/iccad2026contest:$ROOT/FloorSet:${PYTHONPATH:-}"
uv run "$EVALUATOR" \
  --data-path ../ \
  --evaluate "$OPTIMIZER" \
  --test-id "$TESTID" \
  --floorplan-output-dir "$FLOORPLAN_DIR" \
  --verbose \
  "${EXTRA_ARGS[@]}"
