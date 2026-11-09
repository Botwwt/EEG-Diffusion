#!/bin/bash
#SBATCH --job-name=eeg_T_geom262
#SBATCH --partition=C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15000M
#SBATCH --time=00:10:00
set -euo pipefail
umask 077
PROJECT=/dfs/share/maleiLab/wwt/eeg
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$PROJECT"
for attempt in 1 2 3 4 5 6; do
if sha256sum --check <<'SOURCE_SHA'
37d90704d72d23e06dbf4f1d56feb595fefa234a7b70b7f07f92bb02e1e36a23  /dfs/share/maleiLab/wwt/eeg/code/PKU_probe_matching_ordinary_T_CPU_20261109_v262.py
SOURCE_SHA
then break; fi
test "$attempt" != 6
sleep 2
done
"$PROJECT/envs/torch230_cu121_v235/bin/python" -B "$PROJECT/code/PKU_probe_matching_ordinary_T_CPU_20261109_v262.py"
