#!/bin/bash
# ordchain.sh <gpu> <sub> [<sub> ...]: per subject, train the ordinary head, apply it to the test conditions, then generate
# three arms (head alone, + low-level start, + low-level start and best of 4).
R=${RAD_ROOT:?set RAD_ROOT}; cd $R/code/rad
export RAD_ROOT=$R HF_HOME=$R/cache/hf TORCH_HOME=$R/cache/torch XDG_CACHE_HOME=$R/cache TMPDIR=$R/tmp HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$1 PYTHONWARNINGS=ignore
shift; L=$R/logs/ordchain.log
for s in "$@"; do
  x=$(printf "%02d" $s)
  [ -f $R/runs/ord/sub-$x/head.pt ] || ../../env/bin/python ordh.py train $s > $R/logs/ord_s$x.log 2>&1
  ../../env/bin/python ordh.py apply $s 2>&1 | grep "apply" >> $L
  B="--run t0_r4/sub-$x/fold-1_seed2025 --save 0 --K 4 --cond_file ord/sub-$x.npy"
  printf "%s\n" "$B --name s${x}_t0_s2025_A1_ord" "$B --name s${x}_t0_s2025_A1_ord_low --init_file sub-$x/vae_rcp_n0.8.npy --strength 0.75" | xargs -P 2 -d "\n" -I{} bash -c "../../env/bin/python gen.py {} 2>&1 | grep -E \" \| pixcorr|Error|Traceback\" >> $L"
  ../../env/bin/python gen.py $B --name s${x}_t0_s2025_A1_ord_low_sel4 --init_file sub-$x/vae_rcp_n0.8.npy --strength 0.75 --nsel 4 2>&1 | grep -E " \| pixcorr|Error|Traceback" >> $L
  echo "CHAIN_DONE $s $(date -u +%H:%M:%S)" >> $L
done
