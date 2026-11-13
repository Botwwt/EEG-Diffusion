#!/bin/bash
R=${RAD_ROOT:?set RAD_ROOT}
tmux new-session -d -s eeg-ms2 "bash $R/code/jobs/ms_screen2.sh 4 '--tag ms_e4 --epochs 4' '--tag ms_e6 --epochs 6' '--tag ms_e8 --epochs 8' '--tag ms_e15 --epochs 15' '--tag ms_e10_ft3 --epochs 10 --ft_epochs 3' '--tag ms_e10_ft10in --epochs 10 --ft_epochs 10 --ft_trunk 0 --ft_lr 5e-4' '--tag ms_e10_l2 --epochs 10 --layers 2' '--tag ms_e10_s2026 --epochs 10 --seed 2026' '--tag ms_e10_lr1e-3 --epochs 10 --lr 1e-3' '--tag ms_e10_lr2.5e-4 --epochs 10 --lr 2.5e-4'"
q=$R/tmp/amp_g5.txt; : > $q
for x in 03 06; do echo "--run t0_r4/sub-$x/fold-1_seed2025 --save 0 --K 4 --cond_file ord/sub-$x.npy --init_file sub-$x/vae_rcp_n0.8.npy --strength 0.75 --amp 2 --name s${x}_t0_s2025_A1_ord_low_amp2" >> $q; done
tmux new-session -d -s eeg-amp5 "bash $R/code/jobs/rg.sh 5 $q 2"
sleep 1; tmux ls
