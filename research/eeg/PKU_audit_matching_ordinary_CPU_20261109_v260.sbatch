#!/bin/bash
#SBATCH --job-name=eeg_O_a260
#SBATCH --partition=C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3500M
#SBATCH --time=00:05:00
set -euo pipefail
umask 077
PROJECT=/dfs/share/maleiLab/wwt/eeg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$PROJECT"
for attempt in 1 2 3 4 5 6; do
if sha256sum --check <<'SOURCE_SHA'
accff394db479def6f11cec48ee272cc4a0d22c104756a45884232f8de284b89  /dfs/share/maleiLab/wwt/eeg/code/PKU_audit_matching_ordinary_CPU_20261109_v260.py
SOURCE_SHA
then break; fi
test "$attempt" != 6
sleep 2
done
"$PROJECT/envs/torch230_cu121_v235/bin/python" -B "$PROJECT/code/PKU_audit_matching_ordinary_CPU_20261109_v260.py"
