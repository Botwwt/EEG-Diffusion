#!/bin/bash
#SBATCH --job-name=eeg_anchor_dispatch293
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
c17a5f043674b22041780e1b727d0eaa4929670296754ba6a286e6bdb81fe94e  /dfs/share/maleiLab/wwt/eeg/code/PKU_dispatch_anchor_CPU_20261109_v293.py
SOURCE_SHA
"$PROJECT/envs/torch230_cu121_v235/bin/python" -B "$PROJECT/code/PKU_dispatch_anchor_CPU_20261109_v293.py" --hop "$1"
