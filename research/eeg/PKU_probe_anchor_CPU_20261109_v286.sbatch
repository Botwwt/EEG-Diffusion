#!/bin/bash
#SBATCH --job-name=eeg_anchor_CPU286
#SBATCH --partition=C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15000M
#SBATCH --time=00:10:00
set -euo pipefail
umask 077
PROJECT=/dfs/share/maleiLab/wwt/eeg
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$PROJECT"
export CUDA_VISIBLE_DEVICES=""
for attempt in 1 2 3 4 5 6; do
if sha256sum --check <<'SOURCE_SHA'
e3fc10c4a6edaaaa3fb663aadcb9ff5fab183f4e9ca339a6da5e931b48230af2  /dfs/share/maleiLab/wwt/eeg/code/PKU_train_anchor_GPU_20261109_v286.py
88e427bf903d29c015702faf89e6cbbffdab3205fbc22dea5b0a0c11229a3fc7  /dfs/share/maleiLab/wwt/eeg/code/PKU_audit_anchor_CPU_20261109_v286.py
29dd126b925aa2878fff725841f34b51da4a7dc438da261e0c2c40f1d50cde6f  /dfs/share/maleiLab/wwt/eeg/code/PKU_queue_anchor_GPU_20261109_v286.py
82987813a6e18781e5ad06c68dbbece99b0301d473b37bd9344536d791399c6b  /dfs/share/maleiLab/wwt/eeg/code/PKU_anchored_encoder_core_20261109_v286.py
57e91d5a0df8caeffc81c963236ca478e788cbd8173e432d59f02b013ebcdd81  /dfs/share/maleiLab/wwt/eeg/code/PKU_probe_anchor_CPU_20261109_v286.py
c20034323f417013cc341ca64e8aa6e5584f7b61784923cbd8fe6cc77638d581  /dfs/share/maleiLab/wwt/eeg/code/EEG_anchored_encoder_study_v286.json
93315b41af26f1191925079ecceddbdf7712371bd348b4b4edf2a780557b5270  /dfs/share/maleiLab/wwt/eeg/code/PKU_native_generator_20261109_v253.py
8a63ad3d5a7b65cf46eb02831cbaa0b1a13b9ed94fc0cedf64fd5f7731f2dee4  /dfs/share/maleiLab/wwt/eeg/code/PKU_HIVE_native_classes_20261109_v239.py
d1b490a05d002a7493560959e202c02ec646a3da1a5e699621e90f965f16550d  /dfs/share/maleiLab/wwt/eeg/code/PKU_eeg_candidate_core_20261109_v266.py
SOURCE_SHA
then break; fi
test "$attempt" != 6
sleep 2
done
"$PROJECT/envs/torch230_cu121_v235/bin/python" -B "$PROJECT/code/PKU_probe_anchor_CPU_20261109_v286.py"
