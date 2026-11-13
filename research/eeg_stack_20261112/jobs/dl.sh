#!/bin/bash
R=${RAD_ROOT:?set RAD_ROOT}; cd $R
A="-c --auto-file-renaming=false --console-log-level=warn --summary-interval=60 --file-allocation=none --max-tries=0 --retry-wait=5 --timeout=30 --connect-timeout=15 --lowest-speed-limit=300K --max-file-not-found=3"
aria2c -i tmp/aria_big.txt -j 3 -x 6 -s 6 $A 2>&1 | tee logs/dl_big2.log; echo BIG_DONE | tee -a logs/dl_big2.log
aria2c -i tmp/aria_images.txt -j 12 -x 1 $A 2>&1 | tee logs/dl_images.log; echo DL_DONE | tee -a logs/dl_images.log
