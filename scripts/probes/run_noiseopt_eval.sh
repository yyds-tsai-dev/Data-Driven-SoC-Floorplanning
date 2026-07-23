#!/bin/bash
# Phase B full-100 A/B: opt-in noise optimization on the direct-v2 DDIM path.
# Scoring env + MAX=3.5 / DDIM=25 / DIRECT_MIN=2.0 (the 1.1540+-0.002 baseline
# config), checkpoint = eval_retrieval_direct_control.pt, treatment adds
# PARTNER_NOISE_OPT=hybrid. Baseline (1.1561/1.1518) is NOT re-run.
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$ROOT/FloorSet/iccad2026contest:$ROOT/FloorSet"
export DIRECT_CKPT="$ROOT/partner/checkpoints/direct_v2_cont/eval_retrieval_direct_control.pt"
# scoring env (matches scripts/probes/run_budget_scan.sh baseline)
export VKILL_OFF=1 PARTNER_PRESCREEN_V=1 PARTNER_NREF=15 \
       PARTNER_OVERSAMPLE=4 PARTNER_TAG_ANCHOR_EXTRA=3
export PARTNER_BUDGET_MAX=3.5 PARTNER_DDIM_STEPS=25 PARTNER_DIRECT_MIN=2.0
# treatment: opt-in hybrid noise optimization (top-M seeds refined, replace-worst)
export PARTNER_NOISE_OPT="${PARTNER_NOISE_OPT:-hybrid}"
export PARTNER_NOPT_TOPM="${PARTNER_NOPT_TOPM:-4}"
export PARTNER_NOPT_ITERS="${PARTNER_NOPT_ITERS:-10}"
export PARTNER_NOPT_UNROLL="${PARTNER_NOPT_UNROLL:-10}"
# Phase A: w_hpwl=0.1 stalls DDIM noise-opt (energy_drop ~0.03 vs 0.28 at 0);
# keep the wirelength term OFF (overlap+boundary only).
export PARTNER_NOISE_OPT_W_HPWL="${PARTNER_NOISE_OPT_W_HPWL:-0}"

OUT="${1:-$ROOT/artifacts/partner_eval/budget35_dm2_noiseopt.json}"
cd "$ROOT/FloorSet/iccad2026contest"
echo "NOISE_OPT=$PARTNER_NOISE_OPT TOPM=$PARTNER_NOPT_TOPM ITERS=$PARTNER_NOPT_ITERS -> $OUT"
uv run python "$ROOT/scripts/iccad2026_evaluate.py" \
  --data-path ../ \
  --evaluate "$ROOT/partner/my_opt_claude.py" \
  --output "$OUT" \
  2>&1 | tail -6

uv run python - "$OUT" << 'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
res = d["test_results"]
rt = sorted(r["runtime_seconds"] for r in res)
tail = [r["runtime_seconds"] for r in res if r["block_count"] >= 100]
p90 = rt[int(0.9 * len(rt))]
feas = sum(1 for r in res if r["is_feasible"])
print("=" * 60)
print(f"no_runtime = {d['total_score_no_runtime']:.4f}   feasible {feas}/100")
print(f"runtime  avg={sum(rt)/len(rt):.2f}s  p90={p90:.2f}s  max={max(rt):.2f}s")
if tail:
    print(f"tail n>=100  avg={sum(tail)/len(tail):.2f}s  max={max(tail):.2f}s  ({len(tail)} cases)")
print("baseline (no noise-opt): 1.1561 / 1.1518  (mean 1.1540 +-0.002)")
print("=" * 60)
EOF
