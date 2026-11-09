#!/bin/bash
#SBATCH --job-name=eeg_anchor_closure289
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
710c2640185628fca02d47f03278cff70ebbe9bfe961e83692185b9e302cfddf  /dfs/share/maleiLab/wwt/eeg/code/PKU_dispatch_anchor_images_CPU_20261109_v289.py
SOURCE_SHA
"$PROJECT/envs/torch230_cu121_v235/bin/python" -B "$PROJECT/code/PKU_dispatch_anchor_images_CPU_20261109_v289.py" --hop "$1"
