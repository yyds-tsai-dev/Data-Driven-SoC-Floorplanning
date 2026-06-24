#!/bin/bash

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EVALUATOR="$ROOT/scripts/iccad2026_evaluate.py"
OPTIMIZER="$ROOT/src/architecture_v11_optimizer.py"
DEFAULT_CKPT="$ROOT/checkpoints/gnn_transformer_best_0521_ns1000000_ep3_encgraph_transformer_h256_l6_acc32_heads8.pt"

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
#   bash scripts/eval_total.sh                         # use FLOORSET_GNN_CHECKPOINT/.env, else current v10 global best
#   bash scripts/eval_total.sh gnn_epoch10.pt          # use checkpoints/gnn_epoch10.pt
#   bash scripts/eval_total.sh checkpoints/model.pt    # use repo-relative checkpoint path
#   bash scripts/eval_total.sh /path/to/model.pt       # use absolute checkpoint path
#   bash scripts/eval_total.sh gnn_epoch10.pt --output eval.json
#   bash scripts/eval_total.sh --output eval.json      # use default checkpoint, custom output
#   bash scripts/eval_total.sh --best-since-0512       # evaluate all dated *best*.pt checkpoints from 0512 onward

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

run_evaluator() {
  export PYTHONPATH="$ROOT/FloorSet/iccad2026contest:$ROOT/FloorSet:${PYTHONPATH:-}"
  cd "$ROOT/FloorSet/iccad2026contest"
  uv run "$EVALUATOR" \
    --data-path ../ \
    --evaluate "$OPTIMIZER" \
    --verbose \
    "$@"
}

checkpoint_date() {
  local base
  base="$(basename "$1")"
  if [[ "$base" =~ _([0-9]{4}) ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
  fi
}

run_best_since_0512() {
  local out_dir="$ROOT/artifacts/eval_v10/best_since_0512"
  mkdir -p "$out_dir"

  seed_existing_best_records "$out_dir"

  local candidates=()
  local ckpt date
  while IFS= read -r ckpt; do
    date="$(checkpoint_date "$ckpt")"
    if [ -n "$date" ] && [ $((10#$date)) -ge 512 ]; then
      candidates+=("$ckpt")
    fi
  done < <(find "$ROOT/checkpoints" -maxdepth 1 -type f -name '*best*.pt' | sort)

  if [ "${#candidates[@]}" -eq 0 ]; then
    echo "No dated best checkpoints found from 0512 onward." >&2
    return 1
  fi

  echo "Evaluating ${#candidates[@]} best checkpoints from 0512 onward with $EVALUATOR"
  for ckpt in "${candidates[@]}"; do
    local name
    name="$(basename "$ckpt" .pt)"
    local output="$out_dir/$name.json"
    if [ "${FLOORSET_EVAL_REFRESH:-0}" != "1" ] && is_valid_result_json "$output"; then
      echo
      echo "===== $name ====="
      echo "Reusing existing result: $output"
      continue
    fi
    echo
    echo "===== $name ====="
    export FLOORSET_GNN_CHECKPOINT="$ckpt"
    export FLOORSET_GNN_CHECKPOINT_SOURCE="cli"
    run_evaluator --output "$output"
  done

  echo
  echo "===== v10 ranking ====="
  uv run python - "$out_dir" <<'PY'
import json
import sys
from pathlib import Path

rows = []
for path in sorted(Path(sys.argv[1]).glob("*.json")):
    data = json.loads(path.read_text())
    summary = data.get("summary", {})
    rows.append(
        {
            "checkpoint": path.stem,
            "total": float(data["total_score"]),
            "no_runtime": float(data.get("total_score_no_runtime", data["total_score"])),
            "feasible": int(summary.get("num_feasible", 0)),
            "avg_runtime": float(summary.get("avg_runtime", 0.0)),
            "path": str(path),
        }
    )

rows.sort(key=lambda r: (r["no_runtime"], r["total"], -r["feasible"]))
print("| Rank | Checkpoint | v10 no-runtime | v10 total | Feasible | Avg runtime |")
print("| ---: | --- | ---: | ---: | ---: | ---: |")
for idx, row in enumerate(rows, 1):
    print(
        f"| {idx} | `{row['checkpoint']}.pt` | {row['no_runtime']:.4f} | "
        f"{row['total']:.4f} | {row['feasible']}/100 | {row['avg_runtime']:.2f}s |"
    )

if rows:
    best = rows[0]
    print()
    print(
        "Best checkpoint: "
        f"{best['checkpoint']}.pt "
        f"(v10 no-runtime {best['no_runtime']:.4f}, total {best['total']:.4f})"
    )
PY
}

seed_existing_best_records() {
  local out_dir="$1"
  copy_existing_result \
    "$ROOT/artifacts/eval_v10/current_env_default_v10.json" \
    "$out_dir/gnn_best_0512_ns500000_ep4_h192_l6_acc32.json"
  copy_existing_result \
    "$ROOT/artifacts/eval_v10/checkpoint_0519_v10.json" \
    "$out_dir/gnn_best_0519_ns1000000_ep3_encmpnn_h256_l6_acc32.json"
}

copy_existing_result() {
  local src="$1"
  local dest="$2"
  if [ -f "$src" ] && is_valid_result_json "$src"; then
    cp "$src" "$dest"
  fi
}

is_valid_result_json() {
  local path="$1"
  [ -s "$path" ] || return 1
  uv run python - "$path" <<'PY' >/dev/null 2>&1
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
if "total_score" not in data or "test_results" not in data:
    raise SystemExit(1)
if len(data["test_results"]) == 0:
    raise SystemExit(1)
PY
}

if [ "${1:-}" = "--best-since-0512" ]; then
  run_best_since_0512
  exit $?
fi

if [ -n "${1:-}" ] && [[ "$1" == --* ]]; then
  if [ -z "${FLOORSET_GNN_CHECKPOINT:-}" ]; then
    export FLOORSET_GNN_CHECKPOINT="$DEFAULT_CKPT"
  fi
  export FLOORSET_GNN_CHECKPOINT="$(resolve_ckpt_path "$FLOORSET_GNN_CHECKPOINT")"
  EXTRA_ARGS=("$@")
elif [ -n "${1:-}" ]; then
  export FLOORSET_GNN_CHECKPOINT="$(resolve_ckpt_path "$1")"
  export FLOORSET_GNN_CHECKPOINT_SOURCE="cli"
  EXTRA_ARGS=("${@:2}")
elif [ -z "${FLOORSET_GNN_CHECKPOINT:-}" ]; then
  export FLOORSET_GNN_CHECKPOINT="$DEFAULT_CKPT"
  EXTRA_ARGS=()
else
  export FLOORSET_GNN_CHECKPOINT="$(resolve_ckpt_path "$FLOORSET_GNN_CHECKPOINT")"
  EXTRA_ARGS=()
fi
export FLOORSET_GNN_CHECKPOINT_SOURCE="${FLOORSET_GNN_CHECKPOINT_SOURCE:-dotenv}"

echo "Using checkpoint: $FLOORSET_GNN_CHECKPOINT"
echo "Using evaluator: $EVALUATOR"
echo "Evaluation diagnostics: cost factors, top score contributors, best/worst cost cases"

run_evaluator "${EXTRA_ARGS[@]}"
