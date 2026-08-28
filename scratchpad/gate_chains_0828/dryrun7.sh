#!/bin/bash
# Dry run 7 (adds PARTNER_FINAL_LEGAL_GUARD, default on): repack from the working tree (flow ckpt = ft0828 300k EMA), fresh extract, fresh Python 3.13 venv from requirements.txt only, official evaluator.
ROOT=/ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning; T=/ldaphome/yyds-tsai-dev/.claude/jobs/06af0e53/tmp; P=$T/pack_test7
rm -rf $P; mkdir -p $P
cd $ROOT && bash scripts/pack_cadc1013.sh $P 2>&1 | tail -3 || { echo PACK_FAIL; exit 1; }
mkdir -p $P/x && tar xzf $P/cadc1013.tar.gz -C $P/x && echo EXTRACT_OK
ls -la $P/x/cadc1013/checkpoints/
ln -sfn /ldaphome/yyds-tsai-dev/.claude/jobs/06af0e53/tmp/pack_test6/.venv_eval $P/.venv_eval; echo VENV_REUSED_FROM_DRYRUN6
$P/.venv_eval/bin/python -c "import torch,scipy,numba,shapely; print('venv torch',torch.__version__,'cuda',torch.cuda.is_available(),'scipy',scipy.__version__,'numba',numba.__version__)"
while pgrep -f "iccad2026_evaluate.py" >/dev/null; do sleep 20; done
cd $ROOT/FloorSet/iccad2026contest
CUDA_VISIBLE_DEVICES=3 MPLCONFIGDIR=/tmp/mpl-pack PYTHONPATH=$ROOT/FloorSet/iccad2026contest:$ROOT/FloorSet $P/.venv_eval/bin/python iccad2026_evaluate.py --data-path ../ --evaluate $P/x/cadc1013/op_wrapper.py --output $P/result.json > $P/run.out 2> $P/run.err
echo "DRYRUN7_EXIT=$?"
grep -h "\[selfcheck\]\|\[pool-fallback\]\|\[flowtail\]\|loaded flow\|flow model unavailable" $P/run.err $P/run.out | head -5; echo "legal-guard fires: $(grep -c "\[legal-guard\]" $P/run.err)"
cd $ROOT && uv run python - $P/result.json <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); s = d["summary"]; rs = d["test_results"]
import math; mx=max(r["block_count"] for r in rs); W=sum(math.exp((r["block_count"]-mx)/12) for r in rs)
print("noRT=%.4f feasible=%s avg_rt=%.3f max_rt=%.3f first_case_rt=%.3f" % (sum(math.exp((r["block_count"]-mx)/12)*r["cost"] for r in rs)/W, s.get("num_feasible"), s.get("avg_runtime",0), max(r["runtime_seconds"] for r in rs), rs[0]["runtime_seconds"]))
PY
md5sum $P/cadc1013.tar.gz; ls -la $P/cadc1013.tar.gz
echo DRYRUN7_DONE
