#!/bin/bash
#SBATCH --job-name=eeg_backup242
#SBATCH --partition=C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3500M
#SBATCH --time=00:05:00
set -euo pipefail
umask 077
PROJECT=/dfs/share/maleiLab/wwt/eeg
verified=0
for attempt in 1 2 3 4 5 6; do
  if sha256sum --check <<'SOURCE_SHA'
263ce0a7b62134d45c0cdc83d3833e359deb49d40da9261e28a129ca1f534a5f  /dfs/share/maleiLab/wwt/eeg/code/PKU_backup_native_models_CPU_20261109_v242.py
SOURCE_SHA
  then verified=1; break; fi
  sleep 2
done
test "$verified" = 1
"$PROJECT/envs/torch230_cu121_v235/bin/python" -B "$PROJECT/code/PKU_backup_native_models_CPU_20261109_v242.py"
