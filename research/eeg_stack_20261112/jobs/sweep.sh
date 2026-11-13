#!/bin/bash
# sweep.sh "<subs>" "<gpus>": queue the semantic sweep on top of head + low-level start, round-robin over idle cards
R=${RAD_ROOT:?set RAD_ROOT}; subs=$1; gpus=($2); n=${#gpus[@]}; i=0
for g in "${gpus[@]}"; do used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits); [ $used -gt 500 ] && { echo "gpu $g busy ($used MiB), abort"; exit 1; }; : > $R/tmp/sw_g$g.txt; done
for s in $subs; do x=$(printf %02d $s)
  B="--run t0_r4/sub-$x/fold-1_seed2025 --save 0 --K 4 --cond_file ord/sub-$x.npy --init_file sub-$x/vae_rcp_n0.8.npy --strength 0.75 --name s${x}_t0_s2025_A1_ord_low"
  for a in "_amp1.5 --amp 1.5" "_amp2 --amp 2" "_amp3 --amp 3" "_cfg1.5 --cfg 1.5" "_cfg2 --cfg 2" "_cfg3 --cfg 3" "_ips1.5 --ip_scale 1.5" "_ips2 --ip_scale 2"; do
    echo "$B$a" >> $R/tmp/sw_g${gpus[$((i % n))]}.txt; i=$((i+1)); done; done
for g in "${gpus[@]}"; do tmux new-session -d -s eeg-sw$g "bash $R/code/jobs/rg.sh $g $R/tmp/sw_g$g.txt 2"; done
sleep 1; tmux ls | grep sw; wc -l $R/tmp/sw_g*.txt | tail -1
