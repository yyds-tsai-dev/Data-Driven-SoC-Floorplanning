#!/bin/bash
# Smoke-test the frozen (PyInstaller) solver binary against the native path.
#
#   bash scripts/smoke_frozen.sh              # cases 0 (n=21), 37 (n=58), 82 (n=103)
#   CASES="0 99" bash scripts/smoke_frozen.sh # custom case list
#
# What it checks, per case:
#   1. frozen binary driven by the OFFICIAL evaluator through
#      scripts/op_wrapper.py (the real per-case-spawn submission contract)
#      -> feasible, cost, cost_no_runtime, runtime
#   2. native solver (src/architecture_v11_optimizer.py) on the same case
#      -> same metrics for the comparison band
#   3. direct spawn of the binary on a captured payload under /usr/bin/time
#      -> CPU% proves the fork worker pool actually runs parallel when frozen
#   4. startup probes (3x) for onedir (+ onefile if built): per-case spawn cost
#
# PASS gate: every frozen case feasible. Cost is SA-noisy (restart portfolio,
# wall-clock budget), so cost deltas are reported and flagged (>20%) but only
# feasibility fails the script.

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UV="${UV:-$HOME/.local/bin/uv}"
command -v "$UV" > /dev/null 2>&1 || UV="uv"
BIN="${MY_OPT_BIN:-$ROOT/scripts/dist/my_optimizer/my_optimizer}"
ONEFILE_BIN="$ROOT/scripts/dist/my_optimizer_onefile"
EVALUATOR="$ROOT/scripts/iccad2026_evaluate.py"
CASES="${CASES:-0 37 82}"
OUTDIR="$ROOT/artifacts/frozen_smoke"
mkdir -p "$OUTDIR"

if [ ! -x "$BIN" ]; then
  echo "ERROR: frozen binary not found/executable: $BIN" >&2
  echo "Run: bash scripts/package_submission.sh" >&2
  exit 1
fi

# Same env-default loading discipline as scripts/eval_single.sh: .env values
# fill in unset variables so frozen and native see identical toggles.
while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in ""|\#*) continue ;; esac
  key="${line%%=*}"
  value="${line#*=}"
  if [ -z "${!key+x}" ]; then export "$key=$value"; fi
done < "$ROOT/.env"
export MY_OPT_BIN="$BIN"
export PYTHONPATH="$ROOT/FloorSet/iccad2026contest:$ROOT/FloorSet:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# 0) Startup probes: the per-case spawn cost the contest pays on every case.
# ---------------------------------------------------------------------------
echo "=== startup probes (3x each) ==="
probe() {
  local exe="$1" label="$2"
  for i in 1 2 3; do
    local t0 t1
    t0=$(date +%s.%N)
    "$exe" --startup-probe > "$OUTDIR/probe_${label}_$i.json" 2> /dev/null
    t1=$(date +%s.%N)
    echo "  $label run$i wall=$(echo "$t1 $t0" | awk '{printf "%.2fs", $1-$2}') $(cat "$OUTDIR/probe_${label}_$i.json")"
  done
}
probe "$BIN" onedir
[ -x "$ONEFILE_BIN" ] && probe "$ONEFILE_BIN" onefile

# ---------------------------------------------------------------------------
# 1) Capture raw payloads for the smoke cases (exact evaluator preprocessing)
#    via a throwaway capture-optimizer, then 2) evaluator runs frozen/native.
# ---------------------------------------------------------------------------
CAPTURE_OPT="$OUTDIR/capture_optimizer.py"
cat > "$CAPTURE_OPT" <<'PYEOF'
# Throwaway: capture the exact solve() inputs the evaluator hands an
# optimizer, dump them in op_wrapper payload format, return a trivial row
# layout (score is irrelevant).
import json, math, os
from iccad2026_evaluate import FloorplanOptimizer

class MyOptimizer(FloorplanOptimizer):
    def solve(self, block_count, area_targets, b2b_connectivity, p2b_connectivity,
              pins_pos, constraints, target_positions=None):
        payload = {
            "block_count": int(block_count),
            "area_targets": area_targets.tolist(),
            "b2b_connectivity": b2b_connectivity.tolist(),
            "p2b_connectivity": p2b_connectivity.tolist(),
            "pins_pos": pins_pos.tolist(),
            "constraints": constraints.tolist(),
            "target_positions": target_positions.tolist() if target_positions is not None else None,
        }
        with open(os.environ["SMOKE_PAYLOAD_OUT"], "w") as fh:
            json.dump(payload, fh)
        out, x = [], 0.0
        for i in range(int(block_count)):
            a = max(float(area_targets[i]), 1.0)
            w = math.sqrt(a)
            out.append((x, 0.0, w, a / w))
            x += w + 1.0
        return out
PYEOF

cd "$ROOT/FloorSet/iccad2026contest"

run_eval() {  # run_eval <optimizer.py> <test_id> <out.json> [extra env...]
  local opt="$1" tid="$2" out="$3"
  "$UV" run "$EVALUATOR" --data-path ../ --evaluate "$opt" --test-id "$tid" \
    --output "$out" > "${out%.json}.log" 2>&1
}

FAIL=0
for tid in $CASES; do
  echo "=== case $tid ==="
  payload="$OUTDIR/payload_$tid.json"
  if [ ! -f "$payload" ]; then
    echo "  capturing payload..."
    SMOKE_PAYLOAD_OUT="$payload" run_eval "$CAPTURE_OPT" "$tid" "$OUTDIR/capture_$tid.json" \
      || { echo "  ERROR: payload capture failed (see $OUTDIR/capture_$tid.log)"; FAIL=1; continue; }
  fi

  echo "  frozen (evaluator -> op_wrapper -> binary spawn)..."
  run_eval "$ROOT/scripts/op_wrapper.py" "$tid" "$OUTDIR/frozen_$tid.json" \
    || { echo "  ERROR: frozen eval failed (see $OUTDIR/frozen_$tid.log)"; FAIL=1; }

  echo "  native (evaluator -> in-process solver)..."
  run_eval "$ROOT/src/architecture_v11_optimizer.py" "$tid" "$OUTDIR/native_$tid.json" \
    || { echo "  ERROR: native eval failed (see $OUTDIR/native_$tid.log)"; FAIL=1; }

  echo "  direct spawn under /usr/bin/time (parallelism check)..."
  /usr/bin/time -v "$BIN" --payload "$payload" \
    > "$OUTDIR/direct_$tid.json" 2> "$OUTDIR/direct_$tid.time" \
    || { echo "  ERROR: direct frozen spawn failed"; FAIL=1; }
  grep -E "\[frozen_entry\]" "$OUTDIR/direct_$tid.time" | sed 's/^/    /'
  grep -E "Percent of CPU|Elapsed \(wall" "$OUTDIR/direct_$tid.time" | sed 's/^/    /'
done

# ---------------------------------------------------------------------------
# 3) Summary table + gates.
# ---------------------------------------------------------------------------
cd "$ROOT"
"$UV" run python - "$OUTDIR" $CASES <<'PYEOF'
import json, sys
outdir, cases = sys.argv[1], sys.argv[2:]
print("\n=== frozen vs native summary ===")
hdr = f"{'case':>4} {'n':>4} | {'feas F/N':>9} | {'cost F':>8} {'cost N':>8} {'d%':>7} | {'cnr F':>8} {'cnr N':>8} | {'rt F':>6} {'rt N':>6}"
print(hdr); print("-" * len(hdr))
fail = False
for tid in cases:
    row = {}
    for kind in ("frozen", "native"):
        try:
            d = json.load(open(f"{outdir}/{kind}_{tid}.json"))
            row[kind] = d["test_results"][0]
        except Exception as exc:
            print(f"{tid:>4}  MISSING {kind} result: {exc}")
            fail = True
    if len(row) < 2:
        continue
    f, n = row["frozen"], row["native"]
    if not f["is_feasible"]:
        fail = True
    dpct = 100.0 * (f["cost"] - n["cost"]) / max(n["cost"], 1e-9)
    flag = "  <-- cost band >20%" if abs(dpct) > 20 else ""
    print(f"{tid:>4} {f['block_count']:>4} | "
          f"{str(f['is_feasible'])[0]}/{str(n['is_feasible'])[0]:>7} | "
          f"{f['cost']:8.4f} {n['cost']:8.4f} {dpct:+6.1f}% | "
          f"{f['cost_no_runtime']:8.4f} {n['cost_no_runtime']:8.4f} | "
          f"{f['runtime_seconds']:6.2f} {n['runtime_seconds']:6.2f}{flag}")
print()
if fail:
    print("SMOKE: FAIL (infeasible frozen case or missing result)")
    sys.exit(1)
print("SMOKE: PASS (all frozen cases feasible; check cost band + CPU% above)")
PYEOF
STATUS=$?
[ "$FAIL" -ne 0 ] && exit 1
exit "$STATUS"
