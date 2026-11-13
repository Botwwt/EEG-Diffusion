#!/bin/bash
# newenc.sh <gpu> <subs...>: for the joint encoder ms_e10_ft10in, train the co-author's head, apply it, generate 3 arms
R=${RAD_ROOT:?set RAD_ROOT}; cd $R/code/rad; T=ms_e10_ft10in
export RAD_ROOT=$R HF_HOME=$R/cache/hf TORCH_HOME=$R/cache/torch XDG_CACHE_HOME=$R/cache TMPDIR=$R/tmp HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$1 PYTHONWARNINGS=ignore ORD_TAG=$T
G=$1; shift; L=$R/logs/newenc.log
for s in "$@"; do x=$(printf "%02d" $s)
  [ -f $R/runs/ord_$T/sub-$x/head.pt ] || ../../env/bin/python ordh.py train $s > $R/logs/ordms_s$x.log 2>&1
  ../../env/bin/python ordh.py apply $s 2>&1 | grep -E "apply|Error|Traceback" >> $L
  B="--run $T/sub-$x/fold-1_seed2025 --save 0 --K 4 --cond_file ord_$T/sub-$x.npy"
  I="--init_file sub-$x/vae_rcp_n0.8.npy --strength 0.75"
  printf "%s\n" "$B --name s${x}_${T}_A1_ord" "$B $I --name s${x}_${T}_A1_ord_low" "$B $I --amp 2 --name s${x}_${T}_A1_ord_low_amp2" | xargs -P 2 -d "\n" -I{} bash -c "../../env/bin/python gen.py {} 2>&1 | grep -E \" \| pixcorr|Error|Traceback\" >> $L"
  echo "SUB_DONE $x gpu $G $(date -u +%H:%M:%S)" >> $L
done
echo "WORKER_DONE gpu $G $(date -u +%H:%M:%S)" >> $L
