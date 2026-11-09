#!/bin/bash
#SBATCH --job-name=eeg_evalenv275
#SBATCH --partition=C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15000M
#SBATCH --time=00:15:00
set -euo pipefail
umask 077
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PROJECT=/dfs/share/maleiLab/wwt/eeg
cd "$PROJECT"
sha256sum --check <<'SOURCE_SHA'
611e97d7fce761499879eb913ffbcd112b52cbc32f8ba9fac09b54af0a28b897  /dfs/share/maleiLab/wwt/eeg/code/PKU_verify_offline_evaluation_CPU_20261109_v275.py
fed165ae1878a8f41d0b5bcd7ce40022aa1527f26f072b6af70e9001ea64b1ac  /dfs/share/maleiLab/wwt/eeg/code/PKU_restore_metric_assets_CPU_20261109_v272.py
SOURCE_SHA
"$PROJECT/envs/torch230_cu121_v235/bin/python" -B "$PROJECT/code/PKU_verify_offline_evaluation_CPU_20261109_v275.py"
