#!/bin/bash
#SBATCH --job-name=eeg_error_parts311
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
for attempt in 1 2 3 4 5 6; do
if sha256sum --check <<'SOURCE_SHA'
58458f9bc27897ffbfbde98611f7d146bca5be11cd339dbc77e6192007d0a994  /dfs/share/maleiLab/wwt/eeg/code/PKU_readout_error_decomposition_CPU_20261109_v311.py
661416c042dd24ebeb31444f49fdd90e9e297d382450d19edaf99f1a7570abe4  /dfs/share/maleiLab/wwt/eeg/code/PKU_conjugate_coordinates_20261109_v310.py
SOURCE_SHA
then break; fi
test "$attempt" != 6
sleep 2
done
"$PROJECT/envs/torch230_cu121_v235/bin/python" -B "$PROJECT/code/PKU_readout_error_decomposition_CPU_20261109_v311.py"
