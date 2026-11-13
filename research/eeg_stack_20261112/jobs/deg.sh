#!/bin/bash
R=${RAD_ROOT:?set RAD_ROOT}; cd $R/code/rad
export RAD_ROOT=$R HF_HOME=$R/cache/hf TORCH_HOME=$R/cache/torch XDG_CACHE_HOME=$R/cache TMPDIR=$R/tmp HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=4 PYTHONWARNINGS=ignore
while tmux has-session -t eeg-g9a 2>/dev/null; do sleep 5; done
../../env/bin/python deg.py 2>&1 | grep -v -i "warn\|UNEXPECTED\|Notes\|^$\|Key \|---\|Loading\|REPORT" >> $R/logs/deg.log
for e in deg4 blur; do for f in -1 0 1 2 3 4; do ../../env/bin/python train.py --sub 8 --seed 2025 --fold $f --tag t7_$e --emb $e 2>&1 | grep -E "^t[0-9a-z_]+ sub-|Error|Traceback" >> $R/logs/deg.log; done; done
echo DEGQ_DONE >> $R/logs/deg.log
