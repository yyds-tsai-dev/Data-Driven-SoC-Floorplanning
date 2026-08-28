#!/bin/bash
# Flow v1 -> tail-tilted fine-tune.  Detached launcher; auto-resumes from
# <ckpt-dir>/latest.pt on any relaunch (a bare re-run of this script resumes).
set -e
ROOT=/ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning
cd "$ROOT"
export PYTHONPATH="$ROOT/FloorSet/iccad2026contest:$ROOT/FloorSet:$ROOT/partner"
export MPLCONFIGDIR=/tmp/floorset-shadow-mpl
export CUDA_VISIBLE_DEVICES=2
export FT_TAIL_TEMP=24
export FT_MAX_WORKER=89
export FT_N_INDEX="$ROOT/artifacts/flow_ft_0828/file_n_index.json"
CK="$ROOT/artifacts/flow_ft_0828/flow_ft0828_ftv1_tailT24_w0-89_lr2e-5_wu1k_bs12_s300k"
exec uv run python artifacts/flow_ft_0828/ft_launch.py \
  --data-path "$ROOT/FloorSet" \
  --checkpoint-dir "$CK" \
  --amp --batch-size 12 --num-workers 8 \
  --lr 2e-5 --warmup 1000 --max-steps 300000 \
  --ema-decay 0.9998 \
  --d-model 640 --layers 14 --heads 10 --node-feat-dim 32 \
  --save-every 5000 --snapshot-every 25000 --keep-recent 3 \
  --gpu-util-cap 0 --vram-fraction 0.5 \
  --hpwl-loss-weight 0.30 --x0-time-weighting none --term-t-prob 0.0
