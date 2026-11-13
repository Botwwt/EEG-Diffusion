#!/bin/bash
# download + prepare the remaining subjects one at a time; the raw 63-channel files are removed after preparation
R=${RAD_ROOT:?set RAD_ROOT}; cd $R; export RAD_ROOT=$R TMPDIR=$R/tmp
for s in 02 04 05 06 07 09 10; do
  [ -f data/prep/sub-${s}_test.npy ] && continue
  D=$R/data/things-eeg/Preprocessed_data_250Hz_whiten/sub-$s; mkdir -p $D
  for f in train.pt test.pt; do
    until aria2c -c -x 8 -s 8 --auto-file-renaming=false --console-log-level=warn --summary-interval=0 --file-allocation=none --max-tries=0 --retry-wait=5 --timeout=30 --connect-timeout=15 --lowest-speed-limit=300K -d $D -o $f https://hf-mirror.com/datasets/Haitao999/things-eeg/resolve/main/Preprocessed_data_250Hz_whiten/sub-$s/$f >> logs/more_subs.log 2>&1; do sleep 3; done
  done
  (cd code/rad && CUDA_VISIBLE_DEVICES= ../../env/bin/python prep.py eeg $((10#$s)) >> $R/logs/more_subs.log 2>&1) && [ -f data/prep/sub-${s}_test.npy ] && rm -f $D/train.pt $D/test.pt
  echo "SUB_DONE $s $(date -u +%H:%M:%S)" >> logs/more_subs.log
done
echo ALL_SUBS_DONE >> logs/more_subs.log
