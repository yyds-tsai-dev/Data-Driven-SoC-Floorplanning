#!/bin/bash

set -euo pipefail

curl -LsSf https://astral.sh/uv/install.sh | sh
mkdir -p test
cp FloorSet/iccad2026contest/optimizer_template.py test/my_optimizer.py
uv run --link-mode=copy FloorSet/iccad2026contest/iccad2026_evaluate.py --evaluate test/my_optimizer.py --test-id 0 --verbose