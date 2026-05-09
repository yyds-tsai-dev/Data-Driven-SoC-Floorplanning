#!/bin/bash

uv run -m floorset_arch.training.train \
  --data-path FloorSet \
  --output-dir checkpoints \
  --num-samples 200000 \
  --val-samples 2000 \
  --epochs 8 \
  --device cuda \
  --hidden-dim 160 \
  --layers 5 \
  --lr 6e-4 \
  --print-every 500 | tee train_arch_v2.log
