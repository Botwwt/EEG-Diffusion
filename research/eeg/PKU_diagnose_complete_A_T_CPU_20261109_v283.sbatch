#!/bin/bash
#SBATCH --job-name=eeg_T_diagnostic283
#SBATCH --partition=C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15000M
#SBATCH --time=00:10:00
set -euo pipefail
umask 077
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PROJECT=/dfs/share/maleiLab/wwt/eeg
cd "$PROJECT"
"$PROJECT/envs/torch230_cu121_v235/bin/python" -B "$PROJECT/code/PKU_diagnose_complete_A_T_CPU_20261109_v283.py"
