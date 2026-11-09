#!/bin/bash
#SBATCH --job-name=eeg_lat_a258
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
export HF_HOME="$PROJECT/package_cache/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
cd "$PROJECT"
for attempt in 1 2 3 4 5 6; do
if sha256sum --check <<'SOURCE_SHA'
f0cc11c780279ff80ef7eeaa86b85264bf1c97796a2c8506ea3e02c6469ef342  /dfs/share/maleiLab/wwt/eeg/code/PKU_audit_native_T_latents_CPU_20261109_v258.py
SOURCE_SHA
then break; fi
test "$attempt" != 6
sleep 2
done
"$PROJECT/envs/torch230_cu121_v235/bin/python" -B "$PROJECT/code/PKU_audit_native_T_latents_CPU_20261109_v258.py"
