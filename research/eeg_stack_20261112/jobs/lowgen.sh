#!/bin/bash
# lowgen.sh <start-image tag> "<gpus>": wait for runs/low/sub-XX/<tag>_n0.8.npy for all subjects, then head + start image
# (single draw) and head + start image + amp 2 + best of 4, on the joint encoder with per-subject heads
R=${RAD_ROOT:?set RAD_ROOT}; T=ms_e10_ft10in; L=$1; gpus=($2); n=${#gpus[@]}; i=0
for s in 01 02 03 04 05 06 07 08 09 10; do while [ ! -f $R/runs/low/sub-$s/${L}_n0.8.npy ]; do sleep 30; done; done
for g in "${gpus[@]}"; do : > $R/tmp/lg_${L}_g$g.txt; done
for x in 01 02 03 04 05 06 07 08 09 10; do B="--run $T/sub-$x/fold-1_seed2025 --save 0 --K 4 --cond_file ord_$T/sub-$x.npy --init_file sub-$x/${L}_n0.8.npy --strength 0.75"
  for a in "--name s${x}_${T}_A1_ord_${L}" "--amp 2 --nsel 4 --name s${x}_${T}_A1_ord_${L}_amp2_sel4"; do echo "$B $a" >> $R/tmp/lg_${L}_g${gpus[$((i % n))]}.txt; i=$((i+1)); done; done
for g in "${gpus[@]}"; do tmux new-session -d -s eeg-lg_${L}_$g "bash $R/code/jobs/rg.sh $g $R/tmp/lg_${L}_g$g.txt 1"; done
