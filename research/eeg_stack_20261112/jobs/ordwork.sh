#!/bin/bash
# ordwork.sh <gpu> <train|wait:SESSION> <sub> [<sub> ...]
# 1) train the listed subjects heads at the same time on this GPU (or wait for a session that is already training them),
# 2) apply them, 3) then keep taking image jobs from tmp/ordjobs.txt (line: "<sub2> <gen.py args>") until none are left.
R=${RAD_ROOT:?set RAD_ROOT}; cd $R/code/rad
export RAD_ROOT=$R HF_HOME=$R/cache/hf TORCH_HOME=$R/cache/torch XDG_CACHE_HOME=$R/cache TMPDIR=$R/tmp HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$1 PYTHONWARNINGS=ignore
G=$1; M=$2; shift 2; L=$R/logs/ordwork.log; mkdir -p $R/tmp/ordlock
if [ "$M" = train ]; then
  for s in "$@"; do x=$(printf "%02d" $s); [ -f $R/runs/ord/sub-$x/head.pt ] || ../../env/bin/python ordh.py train $s > $R/logs/ord_s$x.log 2>&1 & done; wait
else
  while tmux has-session -t ${M#wait:} 2>/dev/null; do sleep 10; done
fi
for s in "$@"; do ../../env/bin/python ordh.py apply $s 2>&1 | grep "apply" >> $L; done
echo "HEADS_DONE gpu $G subs $* $(date -u +%H:%M:%S)" >> $L
while true; do
  pending=0; ran=0; i=0
  while read -r sub args; do
    i=$((i+1)); [ -d $R/tmp/ordlock/$i ] && continue
    if [ ! -f $R/runs/cond/ord/sub-$sub.npy ]; then pending=1; continue; fi
    mkdir $R/tmp/ordlock/$i 2>/dev/null || continue
    ../../env/bin/python gen.py $args 2>&1 | grep -E " \| pixcorr|Error|Traceback" >> $L; ran=1
  done < $R/tmp/ordjobs.txt
  [ $pending = 0 ] && [ $ran = 0 ] && break
  [ $ran = 0 ] && sleep 20
done
echo "WORKER_DONE gpu $G $(date -u +%H:%M:%S)" >> $L
