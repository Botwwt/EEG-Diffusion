#!/bin/bash
#SBATCH --job-name=eeg_pres_wait317
#SBATCH --partition=C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15000M
#SBATCH --time=00:20:00
set -euo pipefail
umask 077
PROJECT=/dfs/share/maleiLab/wwt/eeg
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=""
cd "$PROJECT"
sha256sum --check <<'SOURCE_SHA'
1484b09d1a0091282ec1f14263ed38cf3fbb5ff32af3e04247b182fe0b6ed420  /dfs/share/maleiLab/wwt/eeg/code/PKU_conditional_preservation_core_20261110_v317.py
1440886db510b30a587e610b7320697a126e1f709c4961634549dcde9aba457f  /dfs/share/maleiLab/wwt/eeg/code/PKU_train_preservation_GPU_20261110_v317.py
23dc0d8f617e7c62562cf7df818df424f560a612ddb791e366e1b279db7ba0b3  /dfs/share/maleiLab/wwt/eeg/code/PKU_audit_preservation_CPU_20261110_v317.py
0fae20f5b1f6685ba5b7279f8d5eccce89fae13efd5a990edc69766aa4285a15  /dfs/share/maleiLab/wwt/eeg/code/PKU_queue_preservation_GPU_20261110_v317.py
43cda0a03a33bca28515608c9a08295d1a6270d8adbd706f9d8741fbf980e127  /dfs/share/maleiLab/wwt/eeg/code/PKU_prepare_preservation_images_CPU_20261110_v317.py
6262f0af7e050ac9b12f32068b059ab6ba9988ed5b5a7516fb54ead2e6957066  /dfs/share/maleiLab/wwt/eeg/code/PKU_generate_preservation_images_GPU_20261110_v317.py
0902ce7e71c4049c693bceb8a863824e79ccd3ea3971a4255c81f457d2166b67  /dfs/share/maleiLab/wwt/eeg/code/PKU_prepare_preservation_scoring_CPU_20261110_v317.py
a3c9825730788320eb4c824bb28be2887a315cf3a8798ec64f67455ac2faf1e2  /dfs/share/maleiLab/wwt/eeg/code/PKU_score_preservation_images_GPU_20261110_v317.py
24f8be43c73355d79f102bfede08c46a6b97a0a2fc43b2a0800918c5584671d1  /dfs/share/maleiLab/wwt/eeg/code/PKU_dispatch_preservation_scoring_CPU_20261110_v317.py
7d1cd6432a950489e4862db698d9fef373ad70e06f3d2c67f1d44e996b33991d  /dfs/share/maleiLab/wwt/eeg/code/PKU_analyze_preservation_scores_CPU_20261110_v317.py
3178d3b5973f48964bc270303f1b78e25a8307de39d9fd3f93954d52a9dfa176  /dfs/share/maleiLab/wwt/eeg/code/PKU_probe_preservation_CPU_20261110_v317.py
6443a92dd84564d3a420852d05fc15883c3e7091874d864c1c4e511ff025ce6b  /dfs/share/maleiLab/wwt/eeg/code/PKU_start_preservation_CPU_20261110_v317.py
e6189dab0ea176e9b457450b7c60be9724f889ce0d8b78c1a063ca575aca3eb7  /dfs/share/maleiLab/wwt/eeg/code/EEG_preservation_policy_v317.json
659c36d5b55994f97474efcac83c2b3fd88fec735b524685370cbba87057312b  /dfs/share/maleiLab/wwt/eeg/code/EEG_preservation_image_policy_v317.json
845f67dae4956ff482e6cd287fc1a686f6fd025e3c1b52c6aee47e78d2bb8504  /dfs/share/maleiLab/wwt/eeg/code/EEG_preservation_scoring_policy_v317.json
cfb8c1edf5d973eecbdb50c11f47bc04458aa370e3e899f2200d289ba6073094  /dfs/share/maleiLab/wwt/eeg/code/EEG_preservation_pins_v317.json
SOURCE_SHA
test "$(df --output=avail -B1 "$PROJECT" | tail -1)" -gt 107374182400
"$PROJECT/envs/torch230_cu121_v235/bin/python" -B "$PROJECT/code/PKU_start_preservation_CPU_20261110_v317.py" --hop "$1"
