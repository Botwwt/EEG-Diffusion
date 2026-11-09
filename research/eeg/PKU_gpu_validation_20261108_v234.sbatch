#!/bin/bash
#SBATCH --job-name=eeg_gpu234
#SBATCH --partition=GPUA800S
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
set -euo pipefail
umask 077
PROJECT=/dfs/share/maleiLab/wwt/eeg_sota_autoresearch_20261018_v1
test -n "${SLURM_JOB_ID:-}"
test "$(readlink -f "$PROJECT")" = "$PROJECT"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PIP_CACHE_DIR="$PROJECT/package_cache"
export HF_HOME="$PROJECT/package_cache/huggingface"
export TORCH_HOME="$PROJECT/package_cache/torch"
export XDG_CACHE_HOME="$PROJECT/package_cache/xdg"
export TRITON_CACHE_DIR="$PROJECT/package_cache/triton"
export CUDA_CACHE_PATH="$PROJECT/package_cache/cuda"
export PYTHONDONTWRITEBYTECODE=1
cd "$PROJECT/code"
exec "$PROJECT/envs/torch230_cu121/bin/python" -B PKU_gpu_validation_20261108_v234.py
