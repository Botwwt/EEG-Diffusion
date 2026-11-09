#!/bin/bash
#SBATCH --job-name=eeg_score_dispatch292
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
32b980dbc83b0f48929a20184ee7d4e3399b20142b1794e4b408ec3f83de8a5a  /dfs/share/maleiLab/wwt/eeg/code/PKU_dispatch_EEG_scoring_CPU_20261109_v292.py
SOURCE_SHA
"$PROJECT/envs/torch230_cu121_v235/bin/python" -B "$PROJECT/code/PKU_dispatch_EEG_scoring_CPU_20261109_v292.py" --hop "$1"
