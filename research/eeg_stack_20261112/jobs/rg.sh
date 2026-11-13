#!/bin/bash
# rg.sh <gpu> <abs queue file> [parallel=2] [script=gen.py]: each line = arguments of the script.
R=${RAD_ROOT:?set RAD_ROOT}; cd $R/code/rad
export RAD_ROOT=$R HF_HOME=$R/cache/hf TORCH_HOME=$R/cache/torch XDG_CACHE_HOME=$R/cache TMPDIR=$R/tmp HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$1 PYTHONWARNINGS=ignore
Q=$2; P=${3:-2}; export S=${4:-gen.py}; export L=$R/logs/$(basename $Q .txt).log
echo "GEN_START $(date -u +%H:%M:%S) gpu $1 $(grep -c . $Q) jobs" >> $L
grep . $Q | xargs -P $P -d "\n" -I{} bash -c "../../env/bin/python \$S {} 2>&1 | grep -E \" \| pixcorr|Error|error|Traceback|assert\" >> \$L"
echo "GEN_DONE $(date -u +%H:%M:%S)" >> $L
