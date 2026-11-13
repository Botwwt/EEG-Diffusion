#!/bin/bash
R=${RAD_ROOT:?set RAD_ROOT}; cd $R
aria2c -i tmp/aria_enc.txt -j 2 -x 8 -s 8 -c --auto-file-renaming=false --console-log-level=warn --summary-interval=60 --file-allocation=none --max-tries=0 --retry-wait=5 --timeout=30 --connect-timeout=15 --lowest-speed-limit=300K 2>&1 | tee logs/dl_enc.log; echo ENC_DONE | tee -a logs/dl_enc.log
