#!/bin/bash
#SBATCH --job-name=eeg_gen_p253
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
export HF_HOME="$PROJECT/package_cache/hf" XDG_CACHE_HOME="$PROJECT/package_cache/xdg"
export TORCH_HOME="$PROJECT/package_cache/torch" TRITON_CACHE_DIR="$PROJECT/package_cache/triton"
export CUDA_CACHE_PATH="$PROJECT/package_cache/cuda" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TMPDIR="$PROJECT/package_cache/tmp_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
cd "$PROJECT"
for attempt in 1 2 3 4 5 6; do
if sha256sum --check <<'SOURCE_SHA'
b6d6322b7970af91a748dafbe779a8e77a1db489446cd6d607e506c36308760f  /dfs/share/maleiLab/wwt/eeg/code/HIVE_adapter_source_v253.py
46f85e7547d8aaa9b91b5ee8f029936f08ad87e4f4bb51f7c6ac3cc69781dcdd  /dfs/share/maleiLab/wwt/eeg/code/HIVE_generator_source_pins_v253.json
2b93c0644bde268ae876f0e2b722106852b25910fedc1cffc97820e96cadf5d7  /dfs/share/maleiLab/wwt/eeg/code/PKU_ordinary_component_source_v253.py
93315b41af26f1191925079ecceddbdf7712371bd348b4b4edf2a780557b5270  /dfs/share/maleiLab/wwt/eeg/code/PKU_native_generator_20261109_v253.py
dafd5c88effbec62f8f9c9832753bfe37b04bcfd49a3cc99a435b551e4ebf2be  /dfs/share/maleiLab/wwt/eeg/code/PKU_profile_native_generator_GPU_20261109_v253.py
e18611d12f546a4d787ddd8fdd1888ccd8ecc3e082c9d91d2d27620bb0a74204  /dfs/share/maleiLab/wwt/eeg/code/native_generator_profile_policy_v253.json
SOURCE_SHA
then break; fi
test "$attempt" != 6
sleep 2
done
"$PROJECT/envs/torch230_cu121_v235/bin/python" -B "$PROJECT/code/PKU_profile_native_generator_GPU_20261109_v253.py"
