#!/bin/bash
# ordw.sh <gpu>: take subjects one at a time (shared lock dir): train the ordinary head (first without gradient
# checkpointing, with it if that runs out of memory), apply it, then generate the two arms (head alone, + low-level start).
R=${RAD_ROOT:?set RAD_ROOT}; cd $R/code/rad
export RAD_ROOT=$R HF_HOME=$R/cache/hf TORCH_HOME=$R/cache/torch XDG_CACHE_HOME=$R/cache TMPDIR=$R/tmp HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$1 PYTHONWARNINGS=ignore
G=$1; L=$R/logs/ordw.log; mkdir -p $R/tmp/ordlock2
one() {  # one <sub>
  s=$1; x=$(printf "%02d" $s)
  if [ ! -f $R/runs/ord/sub-$x/head.pt ]; then
    ORD_CKPT=0 ../../env/bin/python ordh.py train $s > $R/logs/ord_s$x.log 2>&1 || ../../env/bin/python ordh.py train $s > $R/logs/ord_s$x.log 2>&1
  fi
  ../../env/bin/python ordh.py apply $s 2>&1 | grep "apply" >> $L
  B="--run t0_r4/sub-$x/fold-1_seed2025 --save 0 --K 4 --cond_file ord/sub-$x.npy"
  printf "%s\n" "$B --name s${x}_t0_s2025_A1_ord" "$B --name s${x}_t0_s2025_A1_ord_low --init_file sub-$x/vae_rcp_n0.8.npy --strength 0.75" | xargs -P 2 -d "\n" -I{} bash -c "../../env/bin/python gen.py {} 2>&1 | grep -E \" \| pixcorr|Error|Traceback\" >> $L"
  echo "SUB_DONE $x gpu $G $(date -u +%H:%M:%S)" >> $L
}
# the two heads that are already training outside this script
if [ $G = 4 ]; then while tmux has-session -t eeg-ord8 2>/dev/null; do sleep 10; done; one 8; fi
if [ $G = 5 ]; then until [ -f $R/runs/ord/sub-06/head.pt ]; do sleep 10; done; tmux kill-session -t eeg-w5 2>/dev/null; sleep 8; one 6; fi
for s in 1 3 2 4 5 7 9 10; do
  mkdir $R/tmp/ordlock2/s$s 2>/dev/null || continue
  one $s
done
echo "WORKER_DONE gpu $G $(date -u +%H:%M:%S)" >> $L
