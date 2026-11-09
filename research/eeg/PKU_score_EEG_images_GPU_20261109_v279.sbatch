#!/bin/bash
#SBATCH --job-name=eeg_scores276
#SBATCH --partition=GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
set -euo pipefail
umask 077
PROJECT=/dfs/share/maleiLab/wwt/eeg
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
cd "$PROJECT"
for attempt in 1 2 3 4 5 6; do
if sha256sum --check <<'SOURCE_SHA'
daae9219f929599edcd527315212c841d601afaa3a29b61a269055ee02d725d0  /dfs/share/maleiLab/wwt/eeg/code/PKU_EEG_evaluation_core_20261109_v279.py
e3c3cd45f311154522c1ffc6e9bb9308b9a15384a09aed07e9d2a871378e5e91  /dfs/share/maleiLab/wwt/eeg/code/PKU_prepare_EEG_scoring_CPU_20261109_v279.py
8a92839a73f16921bdf24c63fc1a777d36850195101bd2b8ca92b216e7d3aa68  /dfs/share/maleiLab/wwt/eeg/code/PKU_score_EEG_images_GPU_20261109_v279.py
082691fc3f2727f2d930b06266595dd67ae51b4022af67c7df4055117581d04d  /dfs/share/maleiLab/wwt/eeg/code/PKU_analyze_EEG_scores_CPU_20261109_v279.py
d1b0ad82f6ff8afa2a87853fad89c38e843302f45aa9f859c6052a5682cb046a  /dfs/share/maleiLab/wwt/eeg/code/PKU_probe_EEG_metric_core_CPU_20261109_v279.py
85d4365136b58a5b4d87679b3694cfcd4f788cffe6edae65c0a886e9e88aef1e  /dfs/share/maleiLab/wwt/eeg/code/PKU_dispatch_EEG_scoring_CPU_20261109_v279.py
ea59145cd6027c027185c57bd43a17f622bc6ec5eded585c847f671e972c6f5a  /dfs/share/maleiLab/wwt/eeg/code/HIVE_original_utils_eval_v279.py
dc8f47073ca9dfbb9f9c53d196b5ae3787d5d571bfd662e74655c38635af15fd  /dfs/share/maleiLab/wwt/eeg/code/EEG_scoring_policy_v279.json
SOURCE_SHA
then break; fi
test "$attempt" != 6
sleep 2
done
"$PROJECT/envs/torch230_cu121_v235/bin/python" -B "$PROJECT/code/PKU_score_EEG_images_GPU_20261109_v279.py" --stage "$1" --metric "$2"
