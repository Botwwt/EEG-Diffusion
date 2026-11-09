#!/bin/bash
#SBATCH --job-name=eeg_hive_a241
#SBATCH --partition=C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15000M
#SBATCH --time=00:15:00
set -euo pipefail
umask 077
PROJECT=/dfs/share/maleiLab/wwt/eeg
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$PROJECT"
sha256sum --check <<'SOURCE_SHA'
a0a061e35ab330f0ce291a44f92f03868de5b56a128bf088cf1b5c93c8d3653c  /dfs/share/maleiLab/wwt/eeg/code/PKU_audit_native_HIVE_rebuild_CPU_20261109_v241.py
SOURCE_SHA
"$PROJECT/envs/torch230_cu121_v235/bin/python" -B "$PROJECT/code/PKU_audit_native_HIVE_rebuild_CPU_20261109_v241.py"
