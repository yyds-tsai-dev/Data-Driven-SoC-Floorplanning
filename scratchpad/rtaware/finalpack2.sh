#!/bin/bash
# After chainK16: build the s16 candidate packages (guard hardening included from the working tree) and dry-run each.
#   pack_test10 = FT2 + s16              pack_test11 = FT2 + k20 + s16
cd /ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning
T=/ldaphome/yyds-tsai-dev/.claude/jobs/06af0e53/tmp
until grep -q CHAINK16_DONE $T/chainK16.log 2>/dev/null; do sleep 30; done
mk(){ sed -e "s/pack_test8/pack_test$1/g" -e "s/DRYRUN8/DRYRUN$1/g" scratchpad/rtaware/dryrun8.sh > scratchpad/rtaware/dryrun$1.sh; }
bash scratchpad/rtaware/apply_pack_variant.sh ft2 s16 || { echo APPLY_FAIL; exit 1; }
mk 10; bash scratchpad/rtaware/dryrun10.sh 2>&1 | tee $T/dryrun10.log | tail -8
bash scratchpad/rtaware/apply_pack_variant.sh ft2 k20 s16 || { echo APPLY_FAIL; exit 1; }
mk 11; bash scratchpad/rtaware/dryrun11.sh 2>&1 | tee $T/dryrun11.log | tail -8
bash scratchpad/rtaware/apply_pack_variant.sh ft2 s16 >/dev/null
cp $T/pack_test10/cadc1013.tar.gz submission/cadc1013_0830_ft2s16_final.tar.gz
cp $T/pack_test11/cadc1013.tar.gz submission/cadc1013_0830_ft2k20s16_final.tar.gz
cp $T/pack_test10/result.json artifacts/shadow/dryrun10_ft2s16_off.json; cp $T/pack_test11/result.json artifacts/shadow/dryrun11_ft2k20s16_off.json
md5sum submission/cadc1013_0830_ft2s16_final.tar.gz submission/cadc1013_0830_ft2k20s16_final.tar.gz
echo FINALPACK2_DONE
