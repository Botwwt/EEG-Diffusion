#!/bin/bash
#SBATCH --job-name=eeg_score_prep281
#SBATCH --partition=C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15000M
#SBATCH --time=00:20:00
set -euo pipefail
umask 077
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PROJECT=/dfs/share/maleiLab/wwt/eeg
cd "$PROJECT"
sha256sum --check <<'SOURCE_SHA'
ab6a60674ecaf68b9fc958c7f2945de4e3cb64b05e5120001b09a45540b9925e  /dfs/share/maleiLab/wwt/eeg/code/PKU_prepare_EEG_scoring_CPU_20261109_v281.py
SOURCE_SHA
"$PROJECT/envs/torch230_cu121_v235/bin/python" -B "$PROJECT/code/PKU_prepare_EEG_scoring_CPU_20261109_v281.py"
