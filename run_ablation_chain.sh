#!/bin/bash
set -e
cd /nashome/NVL4/vdalab/yyds-dev/codex-worktrees/flow-matching-f1-f3
export PYTHONPATH="$PWD/FloorSet/iccad2026contest:$PWD/FloorSet:$PWD/partner"
export PATH="$HOME/.local/bin:$PATH"
COMMON="--data-path FloorSet --max-steps 200000 --seed 4321 \
  --batch-size 12 --d-model 640 --layers 14 --heads 10 --node-feat-dim 32 \
  --lr 8e-5 --warmup 4000 --ema-decay 0.9998 --amp \
  --vram-fraction 0.85 --gpu-util-cap 0.95 \
  --save-every 1000 --snapshot-every 25000 --keep-recent 3 --log-every 50 \
  --x0-time-weighting none"
echo "=== Arm A (variance floor, term-t off) $(date) ==="
uv run python partner/flow_train_claude.py $COMMON \
  --checkpoint-dir checkpoints/flow_v2_1_armA --term-t-prob 0.0
echo "=== Arm B (terminal-t only) $(date) ==="
uv run python partner/flow_train_claude.py $COMMON \
  --checkpoint-dir checkpoints/flow_v2_1_armB --term-t-prob 0.10 --term-band 0.02
echo "=== ablation chain complete $(date) ==="
