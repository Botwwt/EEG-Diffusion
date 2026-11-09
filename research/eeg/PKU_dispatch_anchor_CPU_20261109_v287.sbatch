#!/bin/bash
#SBATCH --job-name=eeg_anchor_dispatch287
#SBATCH --partition=C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3500M
#SBATCH --time=00:05:00
set -euo pipefail
umask 077
export CUDA_VISIBLE_DEVICES=""
PROJECT=/dfs/share/maleiLab/wwt/eeg
cd "$PROJECT"
sha256sum --check <<'SOURCE_SHA'
94d1545d5c1c6f350a1074f2d86ea9ba88a62021c22f9540035e9348f9833ae9  /dfs/share/maleiLab/wwt/eeg/code/PKU_dispatch_anchor_CPU_20261109_v287.py
SOURCE_SHA
"$PROJECT/envs/torch230_cu121_v235/bin/python" -B "$PROJECT/code/PKU_dispatch_anchor_CPU_20261109_v287.py" --hop "$1"
