#!/bin/bash
# Assemble the contest package cadc1013.tar.gz from the CURRENT partner tree.
# Usage: bash scripts/pack_cadc1013.sh <out_dir>
#   out_dir/cadc1013/  and  out_dir/cadc1013.tar.gz
# Contents: import closure of partner/contest_optimizer.py (shipped as op_src.py),
# tests/synth_instances.py (op_wrapper JIT warm-up), partner/shipping/op_wrapper.py,
# partner/shipping/requirements.txt, checkpoints (flow v1 + v2 student as direct).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:?out_dir}"; PK="$OUT/cadc1013"
rm -rf "$PK"; mkdir -p "$PK/checkpoints"
for m in $(cd "$ROOT" && uv run python scripts/probes/package_closure.py 2>/dev/null); do
  [ -f "$ROOT/partner/$m.py" ] && cp "$ROOT/partner/$m.py" "$PK/$m.py"
done
cp "$ROOT/partner/contest_optimizer.py" "$PK/op_src.py"
cp "$ROOT/tests/synth_instances.py" "$PK/synth_instances.py"
cp "$ROOT/partner/shipping/op_wrapper.py" "$PK/op_wrapper.py"
cp "$ROOT/partner/shipping/requirements.txt" "$PK/requirements.txt"
cp "$ROOT/submission/cadc1013/checkpoints/flow_matching_v1_final.pt" "$PK/checkpoints/flow_matching_v1_final.pt"
cp "$ROOT/artifacts/icdc_topology/checkpoints_s2_20k/best.pt" "$PK/checkpoints/direct_v2_student_s2.pt"
rm -rf "$PK"/__pycache__
find "$PK" -type f -exec chmod 644 {} +; find "$PK" -type d -exec chmod 755 {} +
if grep -rn "/ldaphome\|/nashome\|/store" "$PK"/*.py; then echo "ABSOLUTE PATH IN PACKAGE" >&2; exit 1; fi
( cd "$OUT" && tar --owner=0 --group=0 --numeric-owner --sort=name -czf cadc1013.tar.gz cadc1013 )
echo "entries: $(tar tzf "$OUT/cadc1013.tar.gz" | wc -l)"; md5sum "$OUT/cadc1013.tar.gz"
