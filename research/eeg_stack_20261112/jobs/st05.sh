#!/bin/bash
# st05.sh "<subs>" "<gpus>": low-level operating point (img2img strength 0.5) with and without best of 4, one job per card at a time
R=${RAD_ROOT:?set RAD_ROOT}; T=ms_e10_ft10in; subs=$1; gpus=($2); n=${#gpus[@]}; i=0
for g in "${gpus[@]}"; do : > $R/tmp/st05_g$g.txt; done
for s in $subs; do x=$(printf %02d $s); B="--run $T/sub-$x/fold-1_seed2025 --save 0 --K 4 --cond_file ord_$T/sub-$x.npy --init_file sub-$x/vae_rcp_n0.8.npy --strength 0.5"
  for a in "--name s${x}_${T}_A1_ord_low_st0.5" "--nsel 4 --name s${x}_${T}_A1_ord_low_st0.5_sel4"; do
    [ -f $R/runs/img/$(echo $a | awk '{print $NF}')/metrics.json ] && continue
    echo "$B $a" >> $R/tmp/st05_g${gpus[$((i % n))]}.txt; i=$((i+1)); done; done
for g in "${gpus[@]}"; do u=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits); [ $u -gt 500 ] && { echo "gpu $g busy, skipped"; continue; }
  tmux new-session -d -s eeg-st05_$g "bash $R/code/jobs/rg.sh $g $R/tmp/st05_g$g.txt 1"; done
sleep 1; tmux ls | grep st05; cat $R/tmp/st05_g*.txt | wc -l
