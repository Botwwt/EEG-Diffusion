#!/bin/bash
# rung.sh <abs queue file>: each line = arguments of gen.py; runs on GPU 5 next to the training queue.
R=${RAD_ROOT:?set RAD_ROOT}; cd $R/code/rad
export RAD_ROOT=$R HF_HOME=$R/cache/hf TORCH_HOME=$R/cache/torch XDG_CACHE_HOME=$R/cache TMPDIR=$R/tmp HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=5 PYTHONWARNINGS=ignore
Q=$1; L=$R/logs/$(basename $Q .txt).log
echo "GEN_START $(date -u +%H:%M:%S) $(wc -l < $Q) jobs" >> $L
while read -r line; do [ -z "$line" ] && continue; s=$(date +%s); ../../env/bin/python gen.py $line 2>&1 | grep -E " \| pixcorr|Error|error|Traceback|assert" >> $L; echo "   took $(( $(date +%s) - s ))s: $line" >> $L; done < $Q
echo "GEN_DONE $(date -u +%H:%M:%S)" >> $L
