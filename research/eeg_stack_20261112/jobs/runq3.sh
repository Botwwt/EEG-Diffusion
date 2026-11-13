#!/bin/bash
# runq3.sh <abs queue file>: training queue on GPU 5.
R=${RAD_ROOT:?set RAD_ROOT}; cd $R/code/rad
export RAD_ROOT=$R HF_HOME=$R/cache/hf TORCH_HOME=$R/cache/torch XDG_CACHE_HOME=$R/cache TMPDIR=$R/tmp HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=5 PYTHONWARNINGS=ignore
Q=$1; L=$R/logs/$(basename $Q .txt).log
echo "QUEUE_START $(date -u +%H:%M:%S) $(grep -c . $Q) jobs" >> $L
while read -r line; do [ -z "$line" ] && continue; ../../env/bin/python train.py $line 2>&1 | grep -E "^t[0-9a-z_]+ sub-|Error|error|Traceback" >> $L; done < $Q
echo "QUEUE_DONE $(date -u +%H:%M:%S)" >> $L
