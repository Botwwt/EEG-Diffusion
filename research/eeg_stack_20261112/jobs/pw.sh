#!/bin/bash
# pw.sh <gpu> <abs queue file> [script=gen.py]: pop one line at a time (flock) and run it; several workers may share a queue
R=${RAD_ROOT:?set RAD_ROOT}; cd $R/code/rad
export RAD_ROOT=$R HF_HOME=$R/cache/hf TORCH_HOME=$R/cache/torch XDG_CACHE_HOME=$R/cache TMPDIR=$R/tmp HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$1 PYTHONWARNINGS=ignore
Q=$2; S=${3:-gen.py}; L=$R/logs/$(basename $Q .txt)_g$1.log
echo "PW_START $(date -u +%H:%M:%S) gpu $1" >> $L
while true; do
  line=$(flock $Q.lock bash -c "l=\$(grep -m1 . $Q); [ -n \"\$l\" ] && sed -i 0,/./{/./d} $Q; echo \"\$l\"")
  [ -z "$line" ] && break
  echo "RUN $(date -u +%H:%M:%S) ${line##*--name }" >> $L
  eval "../../env/bin/python $S $line" 2>&1 | grep -E " \| pixcorr|Error|error|Traceback|assert" >> $L
done
echo "PW_DONE $(date -u +%H:%M:%S)" >> $L
