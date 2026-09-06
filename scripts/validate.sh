#!/bin/bash
# Official submission-interface validation (no scoring) of the contest package.
#   bash scripts/validate.sh [package_dir]
set -euo pipefail
source "$(dirname "$0")/eval_common.sh"
PKG="$(resolve_package "${1:-}")"
enter_contest_dir
uv run python iccad2026_evaluate.py --validate "$PKG/op_wrapper.py"
