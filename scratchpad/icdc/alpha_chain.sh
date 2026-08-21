#!/bin/bash
# alpha fidelity curve: 6 arms interleaved per rep so session drift hits all
# arms equally.  c = no injection (drift anchor); a00..a75 = interpolated;
# o = alpha 1 re-run in-session (removes the cross-session drift on 1.0932).
set -uo pipefail
S="/tmp/claude-1100/-nashome-NVL4-vdalab-yyds-dev-Data-Driven-SoC-Floorplanning/c70131c1-0951-4c49-9b1e-f46a14d6ebfc/scratchpad"
for rep in 1 2; do
  for arm in c a00 a25 a50 a75 o; do
    bash "$S/alpha_arm.sh" "$arm" "$rep"
  done
done
# diagnostic reps with the candidate dump (buffering perturbs runtime -> NOT
# used for the score judgement, only for the direct-win counts)
for arm in c a00 a25 a50 a75 o; do
  bash "$S/alpha_arm.sh" "$arm" 9 "$S/alpha_psel"
done
echo "ALPHA_CHAIN_DONE"
