#!/bin/bash
# Round-2 flow fine-tune: continuation of the round-1 300k tail-tilted EMA
# (shipped FLOW_CKPT) at the evaluator-matched tilt temperature T=12.
# Detached launcher; auto-resumes from <ckpt-dir>/latest.pt on any relaunch.
# Extra CLI args are appended and override (argparse takes the last occurrence),
# e.g. `bash run_ft.sh --max-steps 200` for a smoke.
set -e
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/FloorSet/iccad2026contest:$ROOT/FloorSet:$ROOT/src/solver"
export MPLCONFIGDIR=/tmp/floorset-shadow-mpl
export CUDA_VISIBLE_DEVICES=0
export FT_TAIL_TEMP=12
export FT_MAX_WORKER=89
export FT_N_INDEX="$ROOT/artifacts/flow_finetune_round1_0828_tailT24_300k/file_n_index.json"
CK="$ROOT/artifacts/flow_finetune_round2_0829_tailT12_250k/flow_ft0829_ft300k_tailT12_lr1e-5_wu1k_bs12_s250k"
exec uv run python scripts/training/flow_finetune/ft_launch.py \
  --data-path "$ROOT/FloorSet" \
  --checkpoint-dir "$CK" \
  --amp --batch-size 12 --num-workers 8 \
  --lr 1e-5 --warmup 1000 --max-steps 250000 \
  --ema-decay 0.9998 \
  --d-model 640 --layers 14 --heads 10 --node-feat-dim 32 \
  --save-every 5000 --snapshot-every 25000 --keep-recent 3 \
  --gpu-util-cap 0 --vram-fraction 0.5 \
  --hpwl-loss-weight 0.30 --x0-time-weighting none --term-t-prob 0.0 "$@"
