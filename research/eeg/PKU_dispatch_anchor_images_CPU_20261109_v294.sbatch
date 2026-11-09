#!/bin/bash
#SBATCH --job-name=eeg_anchor_closure294
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
c9b229263fac0eb7ce1fa5d6920883a80097387f19b04c3b6943c236df2fe874  /dfs/share/maleiLab/wwt/eeg/code/PKU_dispatch_anchor_images_CPU_20261109_v294.py
SOURCE_SHA
"$PROJECT/envs/torch230_cu121_v235/bin/python" -B "$PROJECT/code/PKU_dispatch_anchor_images_CPU_20261109_v294.py" --hop "$1"
