#!/bin/bash
# Repeat the refiner-adjacent suites N times and report the summary line plus
# any FAILED test ids -- used to separate a real regression from the timing
# flakes these deadline-gated tests already have.
N=${1:-5}
for i in $(seq 1 "$N"); do
  echo "--- run $i ---"
  uv run pytest tests/test_partner_anytime_ladder.py \
    tests/test_partner_early_exit.py tests/test_partner_refine_stall.py \
    tests/test_partner_csa_refine.py tests/test_refine_invariants.py \
    tests/test_partner_candidate_supply.py -q 2>&1 \
    | grep -E "^FAILED|passed|failed" | tail -5
done
