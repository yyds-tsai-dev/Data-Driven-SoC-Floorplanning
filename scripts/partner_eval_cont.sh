#!/bin/bash
# 一鍵評測 direct_v2 續訓 checkpoint(T11 配置,對照 800k 基線)
# 用法: bash scripts/partner_eval_cont.sh [tag]
#   tag 預設 = 當前步數;快照 cont/latest.pt 後跑全量 100 案
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$ROOT/FloorSet/iccad2026contest:$ROOT/FloorSet"

CONT="$ROOT/partner/checkpoints/direct_v2_cont/latest.pt"
STEP=$(uv run python -c "import torch; print(torch.load('$CONT', map_location='cpu', weights_only=False)['step'])" 2>/dev/null | tail -1)
TAG="${1:-step${STEP}}"
SNAP="$ROOT/partner/checkpoints/direct_v2_cont/eval_${TAG}.pt"
cp "$CONT" "$SNAP"
echo "snapshot: $SNAP (step $STEP)"

export DIRECT_CKPT="$SNAP"
export VKILL_OFF=1 PARTNER_PRESCREEN_V=1 PARTNER_NREF=15 \
       PARTNER_OVERSAMPLE=4 PARTNER_TAG_ANCHOR_EXTRA=3

cd "$ROOT/FloorSet/iccad2026contest"
uv run python "$ROOT/scripts/iccad2026_evaluate.py" \
  --data-path ../ \
  --evaluate "$ROOT/src/solver/contest_optimizer.py" \
  --verbose \
  --output "$ROOT/artifacts/partner_eval/cont_${TAG}.json" \
  2>&1 | tail -20

uv run python - << EOF
import json, math
d = json.load(open("$ROOT/artifacts/partner_eval/cont_${TAG}.json"))
print()
print("=" * 60)
print("續訓 checkpoint (step $STEP)  no-runtime = %.4f" % d["total_score_no_runtime"])
print("對照:800k 基線(T11 配置)= 1.1163 / 1.1181;原始基線 1.1263")
print("=" * 60)
EOF
