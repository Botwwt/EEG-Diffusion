#!/bin/bash
#SBATCH --job-name=eeg_tv255
#SBATCH --partition=C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3500M
#SBATCH --time=00:10:00
set -euo pipefail
umask 077
PROJECT=/dfs/share/maleiLab/wwt/eeg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PIP_CACHE_DIR="$PROJECT/package_cache"
cd "$PROJECT"
for attempt in 1 2 3 4 5 6; do
if sha256sum --check <<'SOURCE_SHA'
90906fd9c7d6ae39c884ce3336453cae09b24dc6eb01e5ea791edcaaf384a102  /dfs/share/maleiLab/wwt/eeg/code/PKU_install_torchvision_CPU_20261109_v255.py
SOURCE_SHA
then break; fi
test "$attempt" != 6
sleep 2
done
"$PROJECT/envs/torch230_cu121_v235/bin/python" -B "$PROJECT/code/PKU_install_torchvision_CPU_20261109_v255.py"
