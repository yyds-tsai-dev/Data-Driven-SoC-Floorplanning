#!/bin/bash

git submodule update --init --recursive
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
uv pip install -r FloorSet/iccad2026contest/requirements.txt
mkdir -p test
cp FloorSet/iccad2026contest/optimizer_template.py test/my_optimizer.py
uv run FloorSet/iccad2026contest/iccad2026_evaluate.py --evaluate test/my_optimizer.py --test-id 0 --verbose