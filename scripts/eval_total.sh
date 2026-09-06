#!/bin/bash
# Full 100-case official evaluation of the contest package.
#
#   bash scripts/eval_total.sh                    # pack src/ -> artifacts/eval_package/, evaluate
#   REPACK=1 bash scripts/eval_total.sh           # force a fresh pack first
#   bash scripts/eval_total.sh <package_dir>      # evaluate an existing unpacked package
#   bash scripts/eval_total.sh -- --quick         # extra evaluator args after --
#
# Writes artifacts/eval_runs/total_<timestamp>.json and prints the no-runtime
# total, feasibility count and runtime tail. Full log: same path with .log.
set -euo pipefail
source "$(dirname "$0")/eval_common.sh"
PKG_ARG=""; [ $# -gt 0 ] && [ "$1" != "--" ] && { PKG_ARG="$1"; shift; }
[ "${1:-}" = "--" ] && shift
PKG="$(resolve_package "$PKG_ARG")"
mkdir -p "$RUNS_DIR"; TS="$(date +%Y%m%d_%H%M%S)"; OUT="$RUNS_DIR/total_$TS.json"
echo "[eval] package: $PKG"; echo "[eval] output : $OUT"
enter_contest_dir
uv run python "$EVALUATOR" --data-path ../ --evaluate "$PKG/op_wrapper.py" --output "$OUT" "$@" 2>&1 | tee "${OUT%.json}.log" | tail -25
echo "legal-guard fires: $(grep -c '\[legal-guard\]' "${OUT%.json}.log" || true)"
summarize_result "$OUT"
