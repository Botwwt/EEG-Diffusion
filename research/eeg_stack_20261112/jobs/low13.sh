#!/bin/bash
R=${RAD_ROOT:?set RAD_ROOT}; cd $R/code/rad
export RAD_ROOT=$R HF_HOME=$R/cache/hf TORCH_HOME=$R/cache/torch XDG_CACHE_HOME=$R/cache TMPDIR=$R/tmp HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=5 PYTHONWARNINGS=ignore
for s in 1 3; do ../../env/bin/python low.py $s 2>&1 | grep -v -i "warn\|OOM" >> $R/logs/low.log; done
bash $R/code/jobs/rg.sh 5 $R/tmp/g9c.txt 2
