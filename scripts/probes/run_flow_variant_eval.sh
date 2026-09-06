#!/bin/bash
# Full-100 A/B for flow-channel noise-opt variants on the D+F 1M st8 pipeline.
# Usage: bash run_flow_variant_eval.sh <baseline|antithetic|zorder|nopt|nopt_rep>
# Control = D+F 1M st8 = 1.1443 / proj 0.8932.
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$ROOT/FloorSet/iccad2026contest:$ROOT/FloorSet"
VARIANT="${1:-baseline}"

# D+F pipeline (promoted) + scoring env + decided budget knobs
export DIRECT_CKPT="$ROOT/partner/checkpoints/direct_v2_cont/eval_retrieval_direct_control.pt"
export FLOW_CKPT="/nashome/NVL4/vdalab/yyds-dev/codex-worktrees/flow-matching-f1-f3/checkpoints/flow_matching_v1/final.pt"
export PARTNER_FLOW_SLOTS=10 PARTNER_FLOW_STEPS=8 PARTNER_FLOW_SOLVER=euler
export VKILL_OFF=1 PARTNER_PRESCREEN_V=1 PARTNER_NREF=15 \
       PARTNER_OVERSAMPLE=4 PARTNER_TAG_ANCHOR_EXTRA=3
export PARTNER_BUDGET_MAX=3.5 PARTNER_DDIM_STEPS=25 PARTNER_DIRECT_MIN=2.0

case "$VARIANT" in
  baseline)   : ;;
  antithetic) export PARTNER_FLOW_ANTITHETIC=1 ;;
  zorder)     export PARTNER_FLOW_ZORDER=1 ;;
  nopt)       export PARTNER_FLOW_NOPT=hybrid ;;
  nopt_rep)   export PARTNER_FLOW_NOPT=hybrid PGUIDE_W_REP="${PGUIDE_W_REP:-0.5}" ;;
  *) echo "unknown variant $VARIANT"; exit 1 ;;
esac

OUT="$ROOT/artifacts/partner_eval/flow_${VARIANT}.json"
cd "$ROOT/FloorSet/iccad2026contest"
echo "VARIANT=$VARIANT -> $OUT"
uv run python "$ROOT/scripts/iccad2026_evaluate.py" \
  --data-path ../ --evaluate "$ROOT/src/solver/contest_optimizer.py" \
  --output "$OUT" 2>&1 | tail -5

uv run python - "$OUT" "$VARIANT" "$ROOT" << 'EOF'
import json, sys
sys.path.insert(0, sys.argv[3] + "/scripts/probes")
from analyze_budget_scan import total
q, proj, rt_sum, feas, tq = total(sys.argv[1])
d = json.load(open(sys.argv[1])); res = d["test_results"]
rt = sorted(r["runtime_seconds"] for r in res)
tail = [r["runtime_seconds"] for r in res if r["block_count"] >= 100]
print("="*60)
print(f"variant={sys.argv[2]}  no_runtime={q:.4f}  proj={proj:.4f}  tailQ={tq:.4f}  feas={feas}/100")
print(f"runtime avg={rt_sum/len(rt):.2f}s p90={rt[int(0.9*len(rt))]:.2f}s max={max(rt):.2f}s"
      + (f"  tail_avg={sum(tail)/len(tail):.2f}s" if tail else ""))
print("control D+F 1M st8: no_runtime 1.1443 / proj 0.8932")
print("="*60)
EOF
