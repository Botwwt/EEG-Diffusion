#!/bin/bash
#SBATCH --job-name=eeg_arch_compression312
#SBATCH --partition=C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15000M
#SBATCH --time=00:20:00
set -euo pipefail
umask 077
PROJECT=/dfs/share/maleiLab/wwt/eeg
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
cd "$PROJECT"
sha256sum --check <<'SOURCE_SHA'
4427c3752585f467b8167f115bac117471a6a8eef4e1f127ed32f2ae64fd9a8d  /dfs/share/maleiLab/wwt/eeg/code/PKU_architecture_decision_CPU_20261109_v312.py
3f3d0106f3f1716ce3c88d795e378108bebf3f95c2b4e0fc0de558498de84289  /dfs/share/maleiLab/wwt/eeg/code/EEG_architecture_decision_v312.json
SOURCE_SHA
"$PROJECT/envs/torch230_cu121_v235/bin/python" -B "$PROJECT/code/PKU_architecture_decision_CPU_20261109_v312.py" --experiment compression
