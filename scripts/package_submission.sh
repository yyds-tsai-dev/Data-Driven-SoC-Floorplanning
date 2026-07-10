#!/bin/bash
# Package the column-backbone solver into a self-contained CPU-only binary
# with PyInstaller, matching the per-case spawn contract in
# scripts/op_wrapper.py (JSON on stdin -> {"positions": ...} on stdout).
#
# Usage:
#   bash scripts/package_submission.sh              # onedir (default, fastest startup)
#   bash scripts/package_submission.sh --onefile    # additionally build the onefile variant
#   bash scripts/package_submission.sh --skip-venv  # reuse existing packaging venv as-is
#
# Output:
#   scripts/dist/my_optimizer/my_optimizer          (onedir; op_wrapper's first candidate)
#   scripts/dist/my_optimizer_onefile               (onefile variant, comparison only)
#
# Why a dedicated packaging venv: the main .venv ships torch 2.6.0+cu124 whose
# libtorch_python.so hard-links libtorch_cuda.so -> ~4.7 GB of nvidia/* libs;
# PyInstaller would have to bundle all of it. The production path
# (FLOORSET_COLUMN_BACKBONE=1) only uses CPU tensor plumbing, so we build from
# a venv with torch 2.6.0+cpu instead. pyinstaller is installed into that
# packaging venv only -- pyproject.toml is NOT modified.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UV="${UV:-$HOME/.local/bin/uv}"
PACK_VENV="$ROOT/build/pack-venv"
PACK_PY="$PACK_VENV/bin/python"
DIST_DIR="$ROOT/scripts/dist"
WORK_DIR="$ROOT/build/pyinstaller"

BUILD_ONEFILE=0
SKIP_VENV=0
for arg in "$@"; do
  case "$arg" in
    --onefile) BUILD_ONEFILE=1 ;;
    --skip-venv) SKIP_VENV=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------------------
# 1) Packaging venv: CPU-only torch + exactly the production deps.
# ---------------------------------------------------------------------------
if [ "$SKIP_VENV" -eq 0 ]; then
  if [ ! -x "$PACK_PY" ]; then
    echo "== creating packaging venv at $PACK_VENV"
    "$UV" venv "$PACK_VENV" --python 3.12
  fi
  echo "== installing CPU-only torch + production deps + pyinstaller"
  "$UV" pip install --python "$PACK_PY" \
    --index-url https://download.pytorch.org/whl/cpu "torch==2.6.0+cpu"
  # Pinned to the main .venv versions (numpy 1.26.4 / shapely 2.0.5) so the
  # frozen solver computes on the same numerics as the native one.
  "$UV" pip install --python "$PACK_PY" \
    "numpy==1.26.4" "shapely==2.0.5" python-dotenv pyinstaller
fi

# Guard: refuse to package a CUDA torch by accident.
TORCH_VER="$("$PACK_PY" -c 'import torch; print(torch.__version__)')"
case "$TORCH_VER" in
  *+cpu) echo "== packaging torch $TORCH_VER (CPU-only, OK)" ;;
  *) echo "ERROR: packaging venv has non-CPU torch ($TORCH_VER); aborting." >&2; exit 1 ;;
esac

# ---------------------------------------------------------------------------
# 2) PyInstaller build (onedir first: per-case spawn pays no unpack cost).
# ---------------------------------------------------------------------------
# Hidden imports: the production chain reaches these through function-level
# imports; PyInstaller's bytecode scan normally finds them, the explicit list
# is belt-and-braces so a future refactor to dynamic imports cannot silently
# drop the legalizer from the bundle.
COMMON_FLAGS=(
  --noconfirm
  --clean
  --paths "$ROOT/src"
  --add-data "$ROOT/.env:."
  --hidden-import floorset_arch.legalizer.column_backbone
  --hidden-import floorset_arch.legalizer.column_slicing
  --hidden-import floorset_arch.refine.api
  --exclude-module tkinter
  --exclude-module matplotlib
  --exclude-module IPython
  --workpath "$WORK_DIR"
  --specpath "$WORK_DIR"
  --distpath "$DIST_DIR"
)

echo "== building onedir bundle"
"$PACK_VENV/bin/pyinstaller" "${COMMON_FLAGS[@]}" \
  --name my_optimizer --onedir "$ROOT/scripts/frozen_entry.py"

if [ "$BUILD_ONEFILE" -eq 1 ]; then
  echo "== building onefile bundle (comparison variant)"
  "$PACK_VENV/bin/pyinstaller" "${COMMON_FLAGS[@]}" \
    --name my_optimizer_onefile --onefile "$ROOT/scripts/frozen_entry.py"
fi

# ---------------------------------------------------------------------------
# 3) Report + minimal self-check (startup probe: imports + fork-pool warmup).
# ---------------------------------------------------------------------------
echo "== bundle sizes"
du -sh "$DIST_DIR/my_optimizer" 2>/dev/null || true
[ -f "$DIST_DIR/my_optimizer_onefile" ] && du -sh "$DIST_DIR/my_optimizer_onefile" || true

echo "== startup probe (onedir)"
"$DIST_DIR/my_optimizer/my_optimizer" --startup-probe

if [ "$BUILD_ONEFILE" -eq 1 ]; then
  echo "== startup probe (onefile)"
  "$DIST_DIR/my_optimizer_onefile" --startup-probe
fi

echo "== done. Binary for op_wrapper: $DIST_DIR/my_optimizer/my_optimizer"
echo "   Smoke test: bash scripts/smoke_frozen.sh"
