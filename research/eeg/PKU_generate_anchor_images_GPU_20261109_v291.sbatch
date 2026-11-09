#!/bin/bash
#SBATCH --job-name=eeg_anchor_images288
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
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$PROJECT"
for attempt in 1 2 3 4 5 6; do
if sha256sum --check <<'SOURCE_SHA'
7cdc380382dd0c8b15e9115064685169f450086b37309fb81ccc556c69bf3189  /dfs/share/maleiLab/wwt/eeg/code/EEG_anchor_image_comparison_v288.json
f9faa0766fb1c25ceb687ba39df194d43ead3ea57c164423b954140c86c843f4  /dfs/share/maleiLab/wwt/eeg/code/PKU_prepare_anchor_images_CPU_20261109_v288.py
d65a5c669aa420ee3c617acc5bd7620bc964b435f0cd82aa35eddf6e582ed4b4  /dfs/share/maleiLab/wwt/eeg/code/PKU_generate_anchor_images_GPU_20261109_v288.py
69416243e9403d078666b9074d1dae40ec154a6b495054c7d80d019a104269e8  /dfs/share/maleiLab/wwt/eeg/code/PKU_prepare_anchor_scoring_CPU_20261109_v288.py
0dc6e6961b86447655181ebca8df3d78353e576188f27dd51fe6a2987a0bde1d  /dfs/share/maleiLab/wwt/eeg/code/PKU_score_anchor_images_GPU_20261109_v288.py
db3b2f11cfac166a7e1e7eb18444a6cc27a376c423c281532750ced037e0ad11  /dfs/share/maleiLab/wwt/eeg/code/PKU_analyze_anchor_scores_CPU_20261109_v288.py
fef21cac02c95786f8f2f6cfb3e9f64bb886e2061199bfac52b353a715ffa2ea  /dfs/share/maleiLab/wwt/eeg/code/EEG_anchor_scoring_policy_v288.json
82987813a6e18781e5ad06c68dbbece99b0301d473b37bd9344536d791399c6b  /dfs/share/maleiLab/wwt/eeg/code/PKU_anchored_encoder_core_20261109_v286.py
daae9219f929599edcd527315212c841d601afaa3a29b61a269055ee02d725d0  /dfs/share/maleiLab/wwt/eeg/code/PKU_EEG_evaluation_core_20261109_v279.py
93315b41af26f1191925079ecceddbdf7712371bd348b4b4edf2a780557b5270  /dfs/share/maleiLab/wwt/eeg/code/PKU_native_generator_20261109_v253.py
28144937c08a66aa81cb6765caf25fa8df3d6f98223e8303c0fc68a56f11dca1  /dfs/share/maleiLab/wwt/eeg/code/PKU_generate_EEG_comparison_GPU_20261109_v290.py
e634327ccfd78592e12221fa9300333a286d77dcefe60b81fc753a52ce90f096  /dfs/share/maleiLab/wwt/eeg/code/PKU_generate_anchor_images_GPU_20261109_v291.py
SOURCE_SHA
then break; fi
test "$attempt" != 6
sleep 2
done
"$PROJECT/envs/torch230_cu121_v235/bin/python" -B "$PROJECT/code/PKU_generate_anchor_images_GPU_20261109_v291.py" --stage "$1"
