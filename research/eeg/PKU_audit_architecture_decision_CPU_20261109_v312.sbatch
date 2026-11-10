#!/bin/bash
#SBATCH --job-name=eeg_arch_replay312
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
8a536ee6d9ffa664acfdaa8f6807d12c6a53d2ea7422ee4c25129bff1c631c46  /dfs/share/maleiLab/wwt/eeg/code/PKU_audit_architecture_decision_CPU_20261109_v312.py
SOURCE_SHA
"$PROJECT/envs/torch230_cu121_v235/bin/python" -B "$PROJECT/code/PKU_audit_architecture_decision_CPU_20261109_v312.py"
