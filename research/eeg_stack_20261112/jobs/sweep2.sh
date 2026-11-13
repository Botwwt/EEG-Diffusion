#!/bin/bash
# sweep2.sh "<subs>" "<gpus>": CLIP / SSIM levers on the joint encoder + per-subject head, one job per card at a time
R=${RAD_ROOT:?set RAD_ROOT}; T=ms_e10_ft10in; subs=$1; gpus=($2); n=${#gpus[@]}; i=0
for g in "${gpus[@]}"; do u=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits); [ $u -gt 500 ] && { echo "gpu $g busy ($u MiB), abort"; exit 1; }; : > $R/tmp/sw2_g$g.txt; done
for s in $subs; do x=$(printf %02d $s)
  B="--run $T/sub-$x/fold-1_seed2025 --save 0 --K 4 --cond_file ord_$T/sub-$x.npy --name s${x}_${T}_A1_ord_low"
  for a in "_loop0.5 --init_file sub-$x/vae_rcp_n0.8.npy --strength 0.75 --loop 0.5" "_loop1 --init_file sub-$x/vae_rcp_n0.8.npy --strength 0.75 --loop 1" \
           "_st0.6 --init_file sub-$x/vae_rcp_n0.8.npy --strength 0.6" "_st0.5 --init_file sub-$x/vae_rcp_n0.8.npy --strength 0.5" \
           "_n1.0 --init_file sub-$x/vae_rcp_n1.0.npy --strength 0.75" "_8st0.5 --init_file sub-$x/vae_rcp_n0.8.npy --strength 0.5 --steps 8"; do
    echo "$B$a" >> $R/tmp/sw2_g${gpus[$((i % n))]}.txt; i=$((i+1)); done; done
for g in "${gpus[@]}"; do tmux new-session -d -s eeg-sw2_$g "bash $R/code/jobs/rg.sh $g $R/tmp/sw2_g$g.txt 1"; done
sleep 1; tmux ls | grep sw2
