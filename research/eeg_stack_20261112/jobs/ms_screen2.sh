#!/bin/bash
# ms_screen2.sh <gpu> <arg string>...: run train_ms.py variants one after another
R=${RAD_ROOT:?set RAD_ROOT}; cd $R/code/rad
export RAD_ROOT=$R HF_HOME=$R/cache/hf TORCH_HOME=$R/cache/torch XDG_CACHE_HOME=$R/cache TMPDIR=$R/tmp HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$1 PYTHONWARNINGS=ignore
L=$R/logs/ms_screen2.log; shift
for a in "$@"; do ../../env/bin/python train_ms.py $a 2>&1 | grep -E 'seed|Error|Traceback|error' | sed 's/ s1:.*| mean/ | mean/' >> $L; done
echo "MS_DONE $(date -u +%H:%M:%S)" >> $L
