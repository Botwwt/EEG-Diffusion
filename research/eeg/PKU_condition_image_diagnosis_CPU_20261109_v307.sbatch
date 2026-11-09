#!/bin/bash
#SBATCH --job-name=eeg_full_diagnosis307
#SBATCH --partition=C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15000M
#SBATCH --time=00:20:00
set -euo pipefail
umask 077
PROJECT=/dfs/share/maleiLab/wwt/eeg
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=""
cd "$PROJECT"
for attempt in 1 2 3 4 5 6; do
if sha256sum --check <<'SOURCE_SHA'
baf84e5b668043fc53315280fd78c1bc3892807de9699389e93572903eb5069a  /dfs/share/maleiLab/wwt/eeg/code/PKU_condition_image_diagnosis_CPU_20261109_v307.py
SOURCE_SHA
then break; fi
test "$attempt" != 6
sleep 2
done
"$PROJECT/envs/torch230_cu121_v235/bin/python" -B "$PROJECT/code/PKU_condition_image_diagnosis_CPU_20261109_v307.py"
