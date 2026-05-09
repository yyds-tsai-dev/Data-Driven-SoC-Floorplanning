#!/bin/bash

uv run -m floorset_arch.training.train \
  --data-path FloorSet \
  --num-samples 1000 \
  --epochs 1 \
  --output-dir checkpoints
