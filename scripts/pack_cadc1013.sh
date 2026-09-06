#!/bin/bash
# Assemble the contest package cadc1013.tar.gz from the CURRENT src/solver tree.
# Usage: bash scripts/pack_cadc1013.sh <out_dir>
#   out_dir/cadc1013/  and  out_dir/cadc1013.tar.gz
# Contents: import closure of src/solver/contest_optimizer.py (shipped as op_src.py),
# src/solver/synth_instances.py (op_wrapper JIT warm-up), src/shipping/op_wrapper.py,
# src/shipping/requirements.txt, checkpoints (flow ft0828-300k EMA + v2 student as direct).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:?out_dir}"; PK="$OUT/cadc1013"
rm -rf "$PK"; mkdir -p "$PK/checkpoints"
for m in $(cd "$ROOT" && uv run python scripts/probes/package_closure.py 2>/dev/null); do
  [ -f "$ROOT/src/solver/$m.py" ] && cp "$ROOT/src/solver/$m.py" "$PK/$m.py"
done
cp "$ROOT/src/solver/contest_optimizer.py" "$PK/op_src.py"
cp "$ROOT/src/solver/synth_instances.py" "$PK/synth_instances.py"
cp "$ROOT/src/shipping/op_wrapper.py" "$PK/op_wrapper.py"
cp "$ROOT/src/shipping/requirements.txt" "$PK/requirements.txt"
# Flow prior = tail-tilted fine-tune of v1, 300k-step cosine anneal, EMA-only export
# (docs .../post-beta-p0-execution.md Sec.17x-17ab; promoted 2026-08-28).
cp "$ROOT/submission/cadc1013/checkpoints/flow_matching_ft0831_soupa50_250k_ema.pt" "$PK/checkpoints/flow_matching_ft0831_soupa50_250k_ema.pt"
cp "$ROOT/artifacts/icdc_topology/checkpoints_s2_20k/best.pt" "$PK/checkpoints/direct_v2_student_s2.pt"
# Optional block-count-routed TAIL flow prior (docs .../post-beta-p0-execution.md
# Sec.17t).  Copied ONLY when FLOW_CKPT_TAIL points at an existing file at pack
# time; op_wrapper ships the two env vars COMMENTED OUT, so a packaged tail
# checkpoint stays inert until the gate promotes it.
if [ -n "${FLOW_CKPT_TAIL:-}" ] && [ -f "${FLOW_CKPT_TAIL:-}" ]; then
  cp "${FLOW_CKPT_TAIL}" "$PK/checkpoints/flow_matching_tail.pt"
  echo "[pack] tail flow ckpt: ${FLOW_CKPT_TAIL:-} -> checkpoints/flow_matching_tail.pt"
elif [ -n "${FLOW_CKPT_TAIL:-}" ]; then
  echo "[pack] WARNING: FLOW_CKPT_TAIL=${FLOW_CKPT_TAIL:-} not found; packing without it" >&2
fi
rm -rf "$PK"/__pycache__
find "$PK" -type f -exec chmod 644 {} +; find "$PK" -type d -exec chmod 755 {} +
if grep -rn "/ldaphome\|/nashome\|/store" "$PK"/*.py; then echo "ABSOLUTE PATH IN PACKAGE" >&2; exit 1; fi
( cd "$OUT" && tar --owner=0 --group=0 --numeric-owner --sort=name -czf cadc1013.tar.gz cadc1013 )
echo "entries: $(tar tzf "$OUT/cadc1013.tar.gz" | wc -l)"; md5sum "$OUT/cadc1013.tar.gz"
