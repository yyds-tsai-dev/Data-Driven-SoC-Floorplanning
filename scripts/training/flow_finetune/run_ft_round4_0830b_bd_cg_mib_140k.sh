#!/bin/bash
# Round-4 flow fine-tune: continuation of the ROUND-2 250k T=12 checkpoint
# (NOT round 3 -- deliberately not stacked on round 3's x0/hpwl tilt) with a
# single variable class changed: the soft-constraint loss weights.
#   boundary 0.3 -> 1.5, cluster 0.3 -> 1.5, mib 0.1 -> 0.5
#   x0 / hpwl / overlap / v held at the v1 values (1.0 / 0.30 / 0.5 / 0.5).
# Detached launcher; auto-resumes from <ckpt-dir>/latest.pt.
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
CK="$ROOT/artifacts/flow_finetune_round4_0830b_bd_cg_mib_140k/flow_ft0830b_ft250k_tailT12_lr5e-6_bd15_cg15_mib05_s140k"
exec uv run python scripts/training/flow_finetune/ft_launch.py \
  --data-path "$ROOT/FloorSet" \
  --checkpoint-dir "$CK" \
  --amp --batch-size 12 --num-workers 8 \
  --lr 5e-6 --warmup 1000 --max-steps 140000 \
  --ema-decay 0.9998 \
  --d-model 640 --layers 14 --heads 10 --node-feat-dim 32 \
  --save-every 5000 --snapshot-every 25000 --keep-recent 3 \
  --gpu-util-cap 0 --vram-fraction 0.5 \
  --v-loss-weight 0.5 --x0-loss-weight 1.0 --overlap-loss-weight 0.5 \
  --hpwl-loss-weight 0.30 \
  --boundary-loss-weight 1.5 --cluster-loss-weight 1.5 --mib-loss-weight 0.5 \
  --x0-time-weighting none --term-t-prob 0.0 "$@"
