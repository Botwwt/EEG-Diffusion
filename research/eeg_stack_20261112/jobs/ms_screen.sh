#!/bin/bash
# wait for the sweep on GPU 4, then screen joint-encoder variants (condition level only; seconds to minutes each)
R=${RAD_ROOT:?set RAD_ROOT}; cd $R/code/rad
export RAD_ROOT=$R HF_HOME=$R/cache/hf TORCH_HOME=$R/cache/torch XDG_CACHE_HOME=$R/cache TMPDIR=$R/tmp HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=4 PYTHONWARNINGS=ignore
L=$R/logs/ms_screen.log
while tmux has-session -t eeg-sw4 2>/dev/null; do sleep 20; done
echo "MS_START $(date -u +%H:%M:%S)" >> $L
for a in "--tag ms_e25" "--tag ms_e10 --epochs 10" "--tag ms_e50 --epochs 50" "--tag ms_e25_ft5 --ft_epochs 5" "--tag ms_e25_ft10in --ft_epochs 10 --ft_trunk 0 --ft_lr 5e-4" "--tag ms_l2 --layers 2"; do
  ../../env/bin/python train_ms.py $a 2>&1 | grep -E 'fold|Error|Traceback|error' >> $L
done
for e in 50 100; do for s in 8 1 5; do ../../env/bin/python train.py --sub $s --epochs $e --tag t9_e$e --save_model 0 2>&1 | grep -E 'seed|Error' >> $L; done; done
echo "MS_DONE $(date -u +%H:%M:%S)" >> $L
