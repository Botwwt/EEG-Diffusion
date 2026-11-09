#!/bin/bash
#SBATCH --job-name=eeg_O_cal264
#SBATCH --partition=GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
set -euo pipefail
umask 077
PROJECT=/dfs/share/maleiLab/wwt/eeg
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HOME="$PROJECT/package_cache/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
cd "$PROJECT"
for attempt in 1 2 3 4 5 6; do
if sha256sum --check <<'SOURCE_SHA'
f66f60a165975fd26d0857739b9a8d4cc55a662e4f1d6bff96c0144f225577d1  /dfs/share/maleiLab/wwt/eeg/code/PKU_calibrate_matching_ordinary_GPU_20261109_v264.py
SOURCE_SHA
then break; fi
test "$attempt" != 6
sleep 2
done
"$PROJECT/envs/torch230_cu121_v235/bin/python" -B "$PROJECT/code/PKU_calibrate_matching_ordinary_GPU_20261109_v264.py"
