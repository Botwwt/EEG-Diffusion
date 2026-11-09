#!/bin/bash
#SBATCH --job-name=eeg_metric_probe276
#SBATCH --partition=C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15000M
#SBATCH --time=00:20:00
set -euo pipefail
umask 077
PROJECT=/dfs/share/maleiLab/wwt/eeg
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
cd "$PROJECT"
export CUDA_VISIBLE_DEVICES=""
for attempt in 1 2 3 4 5 6; do
if sha256sum --check <<'SOURCE_SHA'
747cf6a02cab742530504446cfa2914278f7bbfd003040b5f86376da7f907072  /dfs/share/maleiLab/wwt/eeg/code/PKU_EEG_evaluation_core_20261109_v276.py
45248b50569c6538321581d866d86e6fdd866a2bf482ea83b86b88d69d066bcc  /dfs/share/maleiLab/wwt/eeg/code/PKU_prepare_EEG_scoring_CPU_20261109_v276.py
30f6a86a099f4e8bfff000b70b191991a6d3ac5ac481169e3007efa88e310a40  /dfs/share/maleiLab/wwt/eeg/code/PKU_score_EEG_images_GPU_20261109_v276.py
88b2e6c625e9032bbb06ec4eff5703fde6c6e7edd44bb6e00a2e995318a1a7f2  /dfs/share/maleiLab/wwt/eeg/code/PKU_analyze_EEG_scores_CPU_20261109_v276.py
7db656456e3a9e93e2c3b6906420c044383a9fed6814ed1d4582d15c4066f120  /dfs/share/maleiLab/wwt/eeg/code/PKU_probe_EEG_metric_core_CPU_20261109_v276.py
76a5e4abf96d4bb256bc255a9949b607904d82e3f98f88650d3642e1a54aa0ff  /dfs/share/maleiLab/wwt/eeg/code/PKU_dispatch_EEG_scoring_CPU_20261109_v276.py
ea59145cd6027c027185c57bd43a17f622bc6ec5eded585c847f671e972c6f5a  /dfs/share/maleiLab/wwt/eeg/code/HIVE_original_utils_eval_v276.py
d38791e4bf0eb55c91760e1b344e8b6ad34b13ebd0f9d07b8132638dcb6eb024  /dfs/share/maleiLab/wwt/eeg/code/EEG_scoring_policy_v276.json
SOURCE_SHA
then break; fi
test "$attempt" != 6
sleep 2
done
"$PROJECT/envs/torch230_cu121_v235/bin/python" -B "$PROJECT/code/PKU_probe_EEG_metric_core_CPU_20261109_v276.py"
