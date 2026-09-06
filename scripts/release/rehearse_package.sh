#!/bin/bash
# Final-submission rehearsal: pack from the working tree, extract the tarball,
# evaluate it with the OFFICIAL evaluator from a CLEAN venv built only from the
# package's own requirements.txt (this is what caught the beta failure: a 0-byte
# requirements.txt left scipy missing on the contest box, see
# docs/experiments/2026-08-21-post-beta-p0-execution.md §16).
#
#   bash scripts/release/rehearse_package.sh [work_dir]     (default artifacts/rehearsal)
#   PYTHON=python3.13 CUDA_VISIBLE_DEVICES=3 bash scripts/release/rehearse_package.sh
#
# Pass criteria used on 2026-08-31 (dry runs 12/14): 100/100 feasible,
# [selfcheck] cuda_available=True, loaded flow model step 250000,
# legal-guard fires 0, first-case runtime < 0.1 s.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
P="${1:-$ROOT/artifacts/rehearsal}"; PY="${PYTHON:-python3}"
rm -rf "$P"; mkdir -p "$P"
bash "$ROOT/scripts/pack_cadc1013.sh" "$P" | tail -3
mkdir -p "$P/x" && tar xzf "$P/cadc1013.tar.gz" -C "$P/x" && echo EXTRACT_OK
ls -la "$P/x/cadc1013/checkpoints/"
"$PY" -m venv "$P/.venv_eval"
"$P/.venv_eval/bin/pip" install -q --upgrade pip
"$P/.venv_eval/bin/pip" install -q -r "$P/x/cadc1013/requirements.txt"
"$P/.venv_eval/bin/python" -c "import torch,scipy,numba,shapely; print('venv torch',torch.__version__,'cuda',torch.cuda.is_available(),'scipy',scipy.__version__,'numba',numba.__version__)"
cd "$ROOT/FloorSet/iccad2026contest"
MPLCONFIGDIR=/tmp/mpl-pack PYTHONPATH="$ROOT/FloorSet/iccad2026contest:$ROOT/FloorSet" \
  "$P/.venv_eval/bin/python" iccad2026_evaluate.py --data-path ../ --evaluate "$P/x/cadc1013/op_wrapper.py" --output "$P/result.json" > "$P/run.out" 2> "$P/run.err" || echo "EVALUATOR_EXIT=$?"
grep -h "\[selfcheck\]\|\[pool-fallback\]\|\[flowtail\]\|loaded flow\|flow model unavailable" "$P/run.err" "$P/run.out" | head -5
echo "legal-guard fires: $(grep -c "\[legal-guard\]" "$P/run.err" || true)"
source "$ROOT/scripts/eval_common.sh"; summarize_result "$P/result.json"
md5sum "$P/cadc1013.tar.gz"; ls -la "$P/cadc1013.tar.gz"
echo REHEARSAL_DONE
