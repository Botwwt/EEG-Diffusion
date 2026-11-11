#!/bin/bash
#SBATCH --job-name=eeg_visit_score320
#SBATCH --partition=GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
set -euo pipefail
umask 077
PROJECT=/dfs/share/maleiLab/wwt/eeg
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$PROJECT"
sha256sum --check <<'SOURCE_SHA'
823648916fc3efe39788c4652616e110612d19931b78474a25c2e35b770137ff  /dfs/share/maleiLab/wwt/eeg/code/EEG_visitation_policy_v320.json
e69b1fd75a43820adc9501ec0b9ec3a15c08dfce146eada7a8f643f4b9b97dd1  /dfs/share/maleiLab/wwt/eeg/code/PKU_conditional_visitation_core_20261110_v320.py
06271275b8e1c77de67f6e5d6c5be258e1813d92752e50610886624f60bb8c7f  /dfs/share/maleiLab/wwt/eeg/code/PKU_probe_visitation_CPU_20261110_v320.py
813443c05fca36064f361e878aa3769dfac70fb4c65d24ed398266d6d7c312f0  /dfs/share/maleiLab/wwt/eeg/code/PKU_build_visitation_bank_GPU_20261110_v320.py
32cc4c3f6a88d92ce7814d111d7b9a4cc574da7292e04f12fe2f5219c0a34e64  /dfs/share/maleiLab/wwt/eeg/code/PKU_audit_visitation_bank_CPU_20261110_v320.py
a45ed5d81f8fc0342b9d146f4c006716250a7174ccdca7927c9e2ff12f50f9a1  /dfs/share/maleiLab/wwt/eeg/code/PKU_train_visitation_GPU_20261110_v320.py
d45cde39fd321c89d5b9fd177b286a6b72e606fdfaf894c33266b9e6374fe8ad  /dfs/share/maleiLab/wwt/eeg/code/PKU_audit_visitation_CPU_20261110_v320.py
c368c226d91cdc952f1df2053a086870640035458026ffb6cf2971ba74c665f8  /dfs/share/maleiLab/wwt/eeg/code/PKU_queue_visitation_GPU_20261110_v320.py
6ba0ebb37f9d5c90da22511e73d833b92d7795177d925bab2a076c6ac8ad3496  /dfs/share/maleiLab/wwt/eeg/code/PKU_prepare_visitation_images_CPU_20261110_v320.py
55c435233745206998461d6b89be6cf5241ee010192684f2669dc8984f00b64f  /dfs/share/maleiLab/wwt/eeg/code/PKU_generate_visitation_images_GPU_20261110_v320.py
60a1732b45c387b996cca12ad54a00e1ee3dc66a8c824d71b44bad29e68c7a62  /dfs/share/maleiLab/wwt/eeg/code/PKU_prepare_visitation_scoring_CPU_20261110_v320.py
b5530f351f3ae3ac91501736d1bc4c994318bebe3a5b5451dd86b3e2c41a4057  /dfs/share/maleiLab/wwt/eeg/code/PKU_score_visitation_images_GPU_20261110_v320.py
38613dcdd2392d48e924c1919397040b6c9b15ccae00336f8d8e96050944b43a  /dfs/share/maleiLab/wwt/eeg/code/PKU_dispatch_visitation_scoring_CPU_20261110_v320.py
6887b6f40871737c6801b588b799bdab977c6482e5db8fd70244e944ffd52763  /dfs/share/maleiLab/wwt/eeg/code/PKU_analyze_visitation_scores_CPU_20261110_v320.py
974aa03bcba5c6a633e911b7e7308bcdf9e9f9d9b71e47ce9b456b4bb178ecbf  /dfs/share/maleiLab/wwt/eeg/code/EEG_visitation_image_policy_v320.json
7474742b6f9c1cde02487edc4aa6181c0c5d62ddb752dcd85f673883dbb80288  /dfs/share/maleiLab/wwt/eeg/code/EEG_visitation_scoring_policy_v320.json
fd9bea8975c9675d96b43a77702f747c81a8bb9e54c9281c8cceaf2e578cad97  /dfs/share/maleiLab/wwt/eeg/code/EEG_visitation_pins_v320.json
SOURCE_SHA
test "$(df --output=avail -B1 "$PROJECT" | tail -1)" -gt 107374182400
"$PROJECT/envs/torch230_cu121_v235/bin/python" -B "$PROJECT/code/PKU_score_visitation_images_GPU_20261110_v320.py" --stage "$1" --metric "$2"
