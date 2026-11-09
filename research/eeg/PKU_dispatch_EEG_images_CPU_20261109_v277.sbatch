#!/bin/bash
#SBATCH --job-name=eeg_dispatch277
#SBATCH --partition=C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3500M
#SBATCH --time=00:05:00
set -euo pipefail
umask 077
PROJECT=/dfs/share/maleiLab/wwt/eeg
export CUDA_VISIBLE_DEVICES=""
cd "$PROJECT"
sha256sum --check <<'SOURCE_SHA'
408361db3a00dbeb3c272ea9b9f2ea53ba763984da1ca606fd987d8e715403b2  /dfs/share/maleiLab/wwt/eeg/code/PKU_dispatch_EEG_images_CPU_20261109_v277.py
SOURCE_SHA
"$PROJECT/envs/torch230_cu121_v235/bin/python" -B "$PROJECT/code/PKU_dispatch_EEG_images_CPU_20261109_v277.py" --hop "$1"
