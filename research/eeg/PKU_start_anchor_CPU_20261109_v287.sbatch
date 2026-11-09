#!/bin/bash
#SBATCH --job-name=eeg_anchor_dispatch287
#SBATCH --partition=C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3500M
#SBATCH --time=00:05:00
set -euo pipefail
umask 077
export CUDA_VISIBLE_DEVICES=""
PROJECT=/dfs/share/maleiLab/wwt/eeg
cd "$PROJECT"
sha256sum --check <<'SOURCE_SHA'
f099c7cec93beef83fa93f03354937bb6fe37596bdad24d86602a28d4eea19ff  /dfs/share/maleiLab/wwt/eeg/code/PKU_start_anchor_CPU_20261109_v287.py
SOURCE_SHA
"$PROJECT/envs/torch230_cu121_v235/bin/python" -B "$PROJECT/code/PKU_start_anchor_CPU_20261109_v287.py"
