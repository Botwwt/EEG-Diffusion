#!/bin/bash
# rungp.sh <abs queue file> [parallel=2]: each line = arguments of gen.py; runs P jobs at a time on GPU 5.
R=${RAD_ROOT:?set RAD_ROOT}; cd $R/code/rad
export RAD_ROOT=$R HF_HOME=$R/cache/hf TORCH_HOME=$R/cache/torch XDG_CACHE_HOME=$R/cache TMPDIR=$R/tmp HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=5 PYTHONWARNINGS=ignore
Q=$1; P=${2:-2}; export L=$R/logs/$(basename $Q .txt).log
echo "GEN_START $(date -u +%H:%M:%S) $(grep -c . $Q) jobs" >> $L
grep . $Q | xargs -P $P -d "\n" -I{} bash -c "../../env/bin/python gen.py {} 2>&1 | grep -E \" \| pixcorr|Error|error|Traceback|assert\" >> \$L"
echo "GEN_DONE $(date -u +%H:%M:%S)" >> $L
