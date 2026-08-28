#!/bin/bash
# Local replay of the ACTUAL beta package (0812) on official 100 with CPU torch (= beta deployment signature), to separate config from data in the hidden-vs-public gap.
T=/ldaphome/yyds-tsai-dev/.claude/jobs/06af0e53/tmp; R=/ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning
while pgrep -f "stress.py|iccad2026_evaluate.py" >/dev/null; do sleep 30; done
cd $R/FloorSet/iccad2026contest
CUDA_VISIBLE_DEVICES="" PARTNER_COORD_POLISH= MPLCONFIGDIR=/tmp/mpl-pack PYTHONPATH=$R/FloorSet/iccad2026contest:$R/FloorSet $T/pack_test6/.venv_eval/bin/python iccad2026_evaluate.py --data-path ../ --evaluate $T/beta0812/cadc1013/op_wrapper.py --output $T/beta0812/result_official_cpu.json > $T/beta0812/run.out 2> $T/beta0812/run.err
echo "BETA0812_EXIT=$?"; tail -8 $T/beta0812/run.out | grep -E "Total|Feasible|Runtime"
