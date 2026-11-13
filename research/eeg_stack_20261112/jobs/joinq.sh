#!/bin/bash
# joinq.sh <queue> <session prefix> <gpu>...: when session <prefix><gpu> has ended and the card is idle, start a pw.sh worker there
R=${RAD_ROOT:?set RAD_ROOT}; Q=$1; P=$2; shift 2; left=("$@")
while [ ${#left[@]} -gt 0 ] && grep -q . $Q; do
  rest=()
  for g in "${left[@]}"; do
    if ! tmux has-session -t "=$P$g" 2>/dev/null && [ $(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits) -lt 500 ]; then
      tmux new-session -d -s eeg-pw$g "bash $R/code/jobs/pw.sh $g $Q"
    else rest+=($g); fi
  done
  left=("${rest[@]}"); sleep 60
done
