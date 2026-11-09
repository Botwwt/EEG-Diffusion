#!/bin/bash
#SBATCH --job-name=eeg_hive_t239
#SBATCH --partition=GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
set -euo pipefail
umask 077
PROJECT=/dfs/share/maleiLab/wwt/eeg
test -n "${SLURM_JOB_ID:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PIP_CACHE_DIR="$PROJECT/package_cache"
export XDG_CACHE_HOME="$PROJECT/package_cache/xdg"
export TORCH_HOME="$PROJECT/package_cache/torch"
export TRITON_CACHE_DIR="$PROJECT/package_cache/triton"
export CUDA_CACHE_PATH="$PROJECT/package_cache/cuda"
export TMPDIR="$PROJECT/package_cache/tmp_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
cd "$PROJECT"
"$PROJECT/envs/torch230_cu121_v235/bin/python" -B "$PROJECT/code/PKU_train_HIVE_native_sub03_GPU_20261109_v239.py"
