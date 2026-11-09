#!/bin/bash
#SBATCH --job-name=eeg_img_a254
#SBATCH --partition=C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3500M
#SBATCH --time=00:10:00
set -euo pipefail
umask 077
PROJECT=/dfs/share/maleiLab/wwt/eeg
cd "$PROJECT"
for attempt in 1 2 3 4 5 6; do
if sha256sum --check <<'SOURCE_SHA'
654bbee01362f19c925e955c82985c157d83f8e401b5c26f8ce7c0fce49b1296  /dfs/share/maleiLab/wwt/eeg/code/PKU_audit_training_images_CPU_20261109_v254.py
SOURCE_SHA
then break; fi
test "$attempt" != 6
sleep 2
done
"$PROJECT/envs/torch230_cu121_v235/bin/python" -B "$PROJECT/code/PKU_audit_training_images_CPU_20261109_v254.py"
