#!/bin/bash
#SBATCH --job-name=eeg_hive_t240
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
test -n "${SLURM_JOB_ID:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PIP_CACHE_DIR="$PROJECT/package_cache"
export XDG_CACHE_HOME="$PROJECT/package_cache/xdg"
export TORCH_HOME="$PROJECT/package_cache/torch"
export TRITON_CACHE_DIR="$PROJECT/package_cache/triton"
export CUDA_CACHE_PATH="$PROJECT/package_cache/cuda"
export TMPDIR="$PROJECT/package_cache/tmp_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
cd "$PROJECT"
verified=0
for attempt in 1 2 3 4 5 6; do
  if sha256sum --check <<'NATIVE_SOURCE_SHA'
560b038708652fac9baa98a1f8f642812eacb7bbb172a56e2c8199ebbdc5b559  /dfs/share/maleiLab/wwt/eeg/code/HIVE_layers_source_v239.py
fc06ed93c1f1f8659605ec561d87e85f91ca72856fe034f4cf8416c2b60b39b1  /dfs/share/maleiLab/wwt/eeg/code/HIVE_brain_source_v239.py
4d1bd2d6fde380c056e9303c2e7a24a4a5479daa81fc23dd94d3ce5954d0da6e  /dfs/share/maleiLab/wwt/eeg/code/HIVE_author_source_pins_v239.json
4f72dcc0d21583817047b5987a5589213abecbb706a8c261431e298650a422b3  /dfs/share/maleiLab/wwt/eeg/code/HIVE_fusion_config_v239.json
14ebb1d729db4bbe4dcf88390c7c7ac44c924836920aff46a3a5f1ec100559cb  /dfs/share/maleiLab/wwt/eeg/code/HIVE_native_sub03_protocol_v239.json
8a63ad3d5a7b65cf46eb02831cbaa0b1a13b9ed94fc0cedf64fd5f7731f2dee4  /dfs/share/maleiLab/wwt/eeg/code/PKU_HIVE_native_classes_20261109_v239.py
89cb024125acc51ad8ca3c96cf6ee130c7d66e3dec646fb8f7324768fc69b9b8  /dfs/share/maleiLab/wwt/eeg/code/PKU_prepare_native_HIVE_CPU_20261109_v239.py
87d9105dfe5d775ed03e95e6d2ee19f305d47fc9136731c0a1881756fe8eed41  /dfs/share/maleiLab/wwt/eeg/code/PKU_train_HIVE_native_sub03_GPU_20261109_v239.py
5639d14dd38232dd7b81bbcd0a247c0244e53743071bbac913e2e71c80923210  /dfs/share/maleiLab/wwt/eeg/code/eeg_margin_matrix_free_20261106_v233.py
8749ece52d075d1a5d806b7deb82a763f688d0bcfef914f21a2dcbda88d98a6c  /dfs/share/maleiLab/wwt/eeg/code/check_eeg_cross_host_gpu_runtime_CPU_20261106_v231.py
9929b6782175c41202eeea8ed724a72a804a8fe0f291bd4a525260eafa8df411  /dfs/share/maleiLab/wwt/eeg/code/eeg_cross_host_gpu_runtime_20261106_v231.py
NATIVE_SOURCE_SHA
  then verified=1; break; fi
  sleep 2
done
test "$verified" = 1
"$PROJECT/envs/torch230_cu121_v235/bin/python" -B "$PROJECT/code/PKU_train_HIVE_native_sub03_GPU_20261109_v239.py"
