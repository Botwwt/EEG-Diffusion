#!/bin/bash
#SBATCH --job-name=eeg_score_dispatch281
#SBATCH --partition=C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3500M
#SBATCH --time=00:05:00
set -euo pipefail
umask 077
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PROJECT=/dfs/share/maleiLab/wwt/eeg
cd "$PROJECT"
sha256sum --check <<'SOURCE_SHA'
f200619e2bb221583f0bfe163959e952c693dbf4209baa1250329837f3324e2b  /dfs/share/maleiLab/wwt/eeg/code/PKU_dispatch_EEG_scoring_CPU_20261109_v281.py
SOURCE_SHA
"$PROJECT/envs/torch230_cu121_v235/bin/python" -B "$PROJECT/code/PKU_dispatch_EEG_scoring_CPU_20261109_v281.py" --hop "$1"
