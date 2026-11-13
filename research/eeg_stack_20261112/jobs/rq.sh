#!/bin/bash
# rq.sh <gpu> <abs queue file>: training queue (lines = train.py arguments), then low.py for the subjects listed after it.
R=${RAD_ROOT:?set RAD_ROOT}; cd $R/code/rad
export RAD_ROOT=$R HF_HOME=$R/cache/hf TORCH_HOME=$R/cache/torch XDG_CACHE_HOME=$R/cache TMPDIR=$R/tmp HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$1 PYTHONWARNINGS=ignore
Q=$2; L=$R/logs/$(basename $Q .txt).log; shift 2
echo "QUEUE_START $(date -u +%H:%M:%S) $(grep -c . $Q) jobs" >> $L
while read -r line; do [ -z "$line" ] && continue; ../../env/bin/python train.py $line 2>&1 | grep -E "^t[0-9a-z_]+ sub-|Error|error|Traceback" >> $L; done < $Q
echo "QUEUE_DONE $(date -u +%H:%M:%S)" >> $L
for s in "$@"; do ../../env/bin/python low.py $s 2>&1 | grep -v -i "warn\|OOM" >> $R/logs/low_all.log; echo "LOW_DONE $s $(date -u +%H:%M:%S)" >> $L; done
