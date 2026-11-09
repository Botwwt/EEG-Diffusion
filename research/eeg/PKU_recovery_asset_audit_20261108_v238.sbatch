#!/bin/bash
#SBATCH --job-name=eeg_fullT238
#SBATCH --partition=C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15000M
#SBATCH --time=00:15:00
set -euo pipefail
umask 077
PROJECT=/dfs/share/maleiLab/wwt/eeg
test -n "${SLURM_JOB_ID:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$PROJECT"
"$PROJECT/envs/torch230_cu121_v235/bin/python" -B "$PROJECT/code/PKU_recovery_asset_audit_20261108_v238.py"
