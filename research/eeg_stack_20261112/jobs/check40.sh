#!/bin/bash
# quick image check of the joint encoder without the head: calibration 0.5 + start image (compare with t0 ..._cal0.5_low)
R=${RAD_ROOT:?set RAD_ROOT}; T=ms_e10_ft10in; q=$R/tmp/chk_ms.txt; : > $q
for x in 05 07 09 10; do echo "--run $T/sub-$x/fold-1_seed2025 --save 0 --K 4 --alpha 0.5 --init_file sub-$x/vae_rcp_n0.8.npy --strength 0.75 --name s${x}_${T}_A1_cal0.5_low" >> $q; done
bash $R/code/jobs/rg.sh $1 $q 2
