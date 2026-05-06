#!/bin/bash

uv run -m floorset_arch.training.train_v1 \
  --data-path FloorSet \
  --num-samples 1000 \
  --epochs 1 \
  --out checkpoints/v1.pt
