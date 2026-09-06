#!/bin/bash
# Official evaluation of ONE validation case (default test_id 95) with the contest package.
#
#   bash scripts/eval_single.sh 99                 # case 99, package from artifacts/eval_package/
#   bash scripts/eval_single.sh 99 <package_dir>   # case 99, existing unpacked package
set -euo pipefail
source "$(dirname "$0")/eval_common.sh"
TESTID="${1:-95}"; PKG="$(resolve_package "${2:-}")"
mkdir -p "$RUNS_DIR"; OUT="$RUNS_DIR/single_${TESTID}_$(date +%Y%m%d_%H%M%S).json"
echo "[eval] package: $PKG"; echo "[eval] output : $OUT"
enter_contest_dir
uv run python "$EVALUATOR" --data-path ../ --evaluate "$PKG/op_wrapper.py" --test-id "$TESTID" --output "$OUT" --verbose
