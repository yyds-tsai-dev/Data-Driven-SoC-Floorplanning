#!/bin/bash
# Shared helpers for eval_total.sh / eval_single.sh / validate.sh.
# The thing under test is always the CONTEST PACKAGE (op_wrapper.py + op_src.py +
# helpers + checkpoints), i.e. exactly what scripts/pack_cadc1013.sh produces
# from src/solver + src/shipping. op_wrapper.py carries the promoted env
# (budget table, checkpoints, PARTNER_* knobs) as setdefaults, so no .env is
# needed here.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVALUATOR="$ROOT/scripts/iccad2026_evaluate.py"      # repo copy: adds Total Score (No Runtime)
PACKAGE_ROOT="${PACKAGE_ROOT:-$ROOT/artifacts/eval_package}"
RUNS_DIR="${RUNS_DIR:-$ROOT/artifacts/eval_runs}"

# resolve_package [dir] -> echoes the directory that holds op_wrapper.py.
# No arg: reuse $PACKAGE_ROOT/cadc1013, packing it first if missing or REPACK=1.
resolve_package() {
  local pkg="${1:-}"
  if [ -n "$pkg" ]; then
    [ -f "$pkg/op_wrapper.py" ] || pkg="$pkg/cadc1013"
    [ -f "$pkg/op_wrapper.py" ] || { echo "no op_wrapper.py under $1" >&2; return 1; }
    printf '%s\n' "$(cd "$pkg" && pwd)"; return 0
  fi
  if [ "${REPACK:-0}" = "1" ] || [ ! -f "$PACKAGE_ROOT/cadc1013/op_wrapper.py" ]; then
    mkdir -p "$PACKAGE_ROOT"
    echo "[eval] packing src/solver -> $PACKAGE_ROOT/cadc1013" >&2
    bash "$ROOT/scripts/pack_cadc1013.sh" "$PACKAGE_ROOT" >&2
  fi
  printf '%s\n' "$PACKAGE_ROOT/cadc1013"
}

enter_contest_dir() {
  export PYTHONPATH="$ROOT/FloorSet/iccad2026contest:$ROOT/FloorSet:${PYTHONPATH:-}"
  export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/floorset-mpl-$USER}"
  cd "$ROOT/FloorSet/iccad2026contest"
}

# summarize_result <result.json>: weighted no-runtime total, feasibility, runtime tail.
summarize_result() {
  uv run python - "$1" <<'PY'
import json, math, sys
d = json.load(open(sys.argv[1])); s = d.get("summary", {}); rs = d["test_results"]
mx = max(r["block_count"] for r in rs)
w = [math.exp((r["block_count"] - mx) / 12) for r in rs]
nort = sum(wi * r["cost"] for wi, r in zip(w, rs)) / sum(w)
print("noRT=%.4f total=%.4f feasible=%s/%d avg_rt=%.3fs max_rt=%.3fs first_case_rt=%.3fs" % (
    d.get("total_score_no_runtime", nort), d.get("total_score", float("nan")),
    s.get("num_feasible", "?"), len(rs), s.get("avg_runtime", 0.0),
    max(r["runtime_seconds"] for r in rs), rs[0]["runtime_seconds"]))
PY
}
