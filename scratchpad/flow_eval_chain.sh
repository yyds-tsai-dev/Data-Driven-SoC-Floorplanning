#!/bin/bash
R=/nashome/NVL4/vdalab/yyds-dev/Data-Driven-SoC-Floorplanning
for V in baseline antithetic zorder nopt; do
  echo "########## $V $(date +%H:%M) ##########"
  bash $R/scripts/probes/run_flow_variant_eval.sh $V 2>&1
done
echo "########## CHAIN DONE $(date +%H:%M) ##########"
