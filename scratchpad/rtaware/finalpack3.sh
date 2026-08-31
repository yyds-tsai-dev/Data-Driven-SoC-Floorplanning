#!/bin/bash
# Final packages from the current tree (guard hardening + runtime micro-opt, FT2 model):
#   pack_test12 = B' (FT2+s16, mid)     pack_test13 = D' (FT2+k20+s16)
cd /ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning
T=/ldaphome/yyds-tsai-dev/.claude/jobs/06af0e53/tmp
while pgrep -f "iccad2026_evaluate.py" >/dev/null; do sleep 20; done
mk(){ sed -e "s/pack_test8/pack_test$1/g" -e "s/DRYRUN8/DRYRUN$1/g" scratchpad/rtaware/dryrun8.sh > scratchpad/rtaware/dryrun$1.sh; }
bash scratchpad/rtaware/apply_pack_variant.sh ft2 s16 || { echo APPLY_FAIL; exit 1; }
mk 12; bash scratchpad/rtaware/dryrun12.sh 2>&1 | tee $T/dryrun12.log | tail -8
bash scratchpad/rtaware/apply_pack_variant.sh ft2 k20 s16 || { echo APPLY_FAIL; exit 1; }
mk 13; bash scratchpad/rtaware/dryrun13.sh 2>&1 | tee $T/dryrun13.log | tail -8
bash scratchpad/rtaware/apply_pack_variant.sh ft2 s16 >/dev/null
cp $T/pack_test12/cadc1013.tar.gz submission/cadc1013_0830b_ft2s16_final.tar.gz
cp $T/pack_test13/cadc1013.tar.gz submission/cadc1013_0830b_ft2k20s16_final.tar.gz
cp $T/pack_test12/result.json artifacts/shadow/dryrun12_ft2s16_off.json; cp $T/pack_test13/result.json artifacts/shadow/dryrun13_ft2k20s16_off.json
md5sum submission/cadc1013_0830b_ft2s16_final.tar.gz submission/cadc1013_0830b_ft2k20s16_final.tar.gz
echo FINALPACK3_DONE
