#!/bin/bash
R=/nashome/NVL4/vdalab/yyds-dev/Data-Driven-SoC-Floorplanning
for TAG in antithetic zorder; do
  echo "########## ${TAG}_rep2 $(date +%H:%M) ##########"
  bash $R/scripts/probes/run_flow_variant_eval.sh $TAG 2>&1
  cp $R/artifacts/partner_eval/flow_${TAG}.json $R/artifacts/partner_eval/flow_${TAG}_rep2.json
done
echo "########## RETEST DONE $(date +%H:%M) ##########"
