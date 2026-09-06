#!/bin/bash
# Round-3 flow fine-tune: continuation of the round-2 250k T=12 tail-tilted run
# with an accuracy tilt on the objective (x0 1.0->2.0, hpwl 0.30->0.60) at half
# the round-2 peak LR.  Detached launcher; auto-resumes from <ckpt-dir>/latest.pt.
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
CK="$ROOT/artifacts/flow_finetune_round3_0830_x0_hpwl_150k/flow_ft0830_ft250k_tailT12_lr5e-6_x02_hp06_s150k"
exec uv run python scripts/training/flow_finetune/ft_launch.py \
  --data-path "$ROOT/FloorSet" \
  --checkpoint-dir "$CK" \
  --amp --batch-size 12 --num-workers 8 \
  --lr 5e-6 --warmup 1000 --max-steps 150000 \
  --ema-decay 0.9998 \
  --d-model 640 --layers 14 --heads 10 --node-feat-dim 32 \
  --save-every 5000 --snapshot-every 25000 --keep-recent 3 \
  --gpu-util-cap 0 --vram-fraction 0.5 \
  --x0-loss-weight 2.0 \
  --hpwl-loss-weight 0.60 --x0-time-weighting none --term-t-prob 0.0 "$@"
