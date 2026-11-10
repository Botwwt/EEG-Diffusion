#!/bin/bash
#SBATCH --job-name=eeg_chart_decision310
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
661416c042dd24ebeb31444f49fdd90e9e297d382450d19edaf99f1a7570abe4  /dfs/share/maleiLab/wwt/eeg/code/PKU_conjugate_coordinates_20261109_v310.py
6cb10f7cfb6221783de6f47e293c058bdfddff2ba573790c90f59b73a7a7f83e  /dfs/share/maleiLab/wwt/eeg/code/PKU_conjugate_target_decision_CPU_20261109_v310.py
6d4adb6c4d505023444e323d0fb85cd6c7c8ba53ee49f793fd828789b333c11b  /dfs/share/maleiLab/wwt/eeg/code/EEG_conjugate_target_decision_v310.json
SOURCE_SHA
then break; fi
test "$attempt" != 6
sleep 2
done
"$PROJECT/envs/torch230_cu121_v235/bin/python" -B "$PROJECT/code/PKU_conjugate_target_decision_CPU_20261109_v310.py"
