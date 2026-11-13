#!/bin/bash
R=${RAD_ROOT:?set RAD_ROOT}
while ! grep -q GEN_DONE $R/logs/g2.log 2>/dev/null; do sleep 10; done
bash $R/code/jobs/rung.sh $R/tmp/g3.txt 2> $R/logs/g3.err
