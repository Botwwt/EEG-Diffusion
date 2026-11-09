#!/bin/bash
#SBATCH --job-name=eeg_env235
#SBATCH --partition=C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15000M
#SBATCH --time=00:30:00
set -euo pipefail
umask 077
PROJECT=/dfs/share/maleiLab/wwt/eeg
test -n "${SLURM_JOB_ID:-}"
test "$(readlink -f "$PROJECT")" = "$PROJECT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PIP_CACHE_DIR="$PROJECT/package_cache"
export HF_HOME="$PROJECT/package_cache/huggingface"
export TORCH_HOME="$PROJECT/package_cache/torch"
export XDG_CACHE_HOME="$PROJECT/package_cache/xdg"
export TRITON_CACHE_DIR="$PROJECT/package_cache/triton"
export CUDA_CACHE_PATH="$PROJECT/package_cache/cuda"
export TMPDIR="$PROJECT/package_cache/tmp_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
BASE=/gpfs/share/software/anaconda/3-2023.09-0/bin/python
ENV="$PROJECT/envs/torch230_cu121_v235"
cd "$PROJECT"
"$BASE" "$PROJECT/code/PKU_verify_offline_wheels_20261108_v235.py"
"$BASE" -m venv "$ENV"
"$ENV/bin/python" -m pip install --disable-pip-version-check --no-index \
  --find-links "$PROJECT/package_cache/wheels_v235" -r "$PROJECT/code/PKU_offline_requirements_20261108_v235.txt"
"$ENV/bin/python" -m pip freeze > "$PROJECT/manifests/environment_${SLURM_JOB_ID}.txt"
"$ENV/bin/python" "$PROJECT/code/PKU_environment_completion_20261108_v235.py"
