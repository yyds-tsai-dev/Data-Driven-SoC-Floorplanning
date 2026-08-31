#!/bin/bash
# After chainFinal: build two candidate packages and dry-run each in the clean py3.13 venv.
#   pack_test8 = FT2 only (round-2 T12 250k EMA, mid table)      pack_test9 = FT2 + k20 tail table
# Leaves the working tree at the FT2-only variant (apply_pack_variant.sh ft2).
cd /ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning
T=/ldaphome/yyds-tsai-dev/.claude/jobs/06af0e53/tmp
until grep -q CHAINFINAL_DONE $T/chainFinal.log 2>/dev/null; do sleep 30; done
bash scratchpad/rtaware/apply_pack_variant.sh ft2 || { echo APPLY_FAIL; exit 1; }
bash scratchpad/rtaware/dryrun8.sh 2>&1 | tee $T/dryrun8.log | tail -8
bash scratchpad/rtaware/apply_pack_variant.sh ft2 k20 || { echo APPLY_FAIL; exit 1; }
sed -e 's/pack_test8/pack_test9/g' -e 's/DRYRUN8/DRYRUN9/g' scratchpad/rtaware/dryrun8.sh > scratchpad/rtaware/dryrun9.sh
bash scratchpad/rtaware/dryrun9.sh 2>&1 | tee $T/dryrun9.log | tail -8
bash scratchpad/rtaware/apply_pack_variant.sh ft2 >/dev/null
echo FINALPACK_DONE
