#!/bin/bash
# postpool2.sh <head name> <gpu>...: wait for the shared head, apply it, generate A, B, A', B' (all best of 4)
R=${RAD_ROOT:?set RAD_ROOT}; cd $R/code/rad; T=ms_e10_ft10in; H=$1; shift; GPUS=("$@"); n=${#GPUS[@]}
export RAD_ROOT=$R HF_HOME=$R/cache/hf TORCH_HOME=$R/cache/torch XDG_CACHE_HOME=$R/cache TMPDIR=$R/tmp HF_HUB_OFFLINE=1 PYTHONWARNINGS=ignore ORD_TAG=$T
L=$R/logs/post_$H.log
while [ ! -f $R/runs/ord_$T/$H/head.pt ]; do sleep 60; done
for s in $(seq 1 10); do CUDA_VISIBLE_DEVICES= ORD_HEAD=$H ../../env/bin/python ordh.py apply $s 2>&1 | grep -E "apply|Error|Traceback" >> $L; done
for g in "${GPUS[@]}"; do : > $R/tmp/post_${H}_g$g.txt; done; i=0
for x in 01 02 03 04 05 06 07 08 09 10; do
  B="--run $T/sub-$x/fold-1_seed2025 --save 0 --K 4 --cond_file ord_${T}_$H/sub-$x.npy --strength 0.75 --nsel 4"
  I="--init_file sub-$x/vae_rcp_n0.8.npy"; J="--init_file sub-$x/lowrk0.5_n0.8.npy"
  for a in "$I --name s${x}_${T}_A1_${H}_low_sel4" "$I --amp 2 --name s${x}_${T}_A1_${H}_low_amp2_sel4" "$J --name s${x}_${T}_A1_${H}_lowrk0.5_sel4" "$J --amp 2 --name s${x}_${T}_A1_${H}_lowrk0.5_amp2_sel4"; do
    echo "$B $a" >> $R/tmp/post_${H}_g${GPUS[$((i % n))]}.txt; i=$((i+1)); done; done
for g in "${GPUS[@]}"; do tmux new-session -d -s eeg-post${H}_$g "bash $R/code/jobs/rg.sh $g $R/tmp/post_${H}_g$g.txt 1"; done
echo "POST_LAUNCHED $(date -u +%H:%M:%S)" >> $L
