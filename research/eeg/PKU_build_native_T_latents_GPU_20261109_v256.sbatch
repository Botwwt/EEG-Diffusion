#!/bin/bash
#SBATCH --job-name=eeg_T_latent256
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
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TMPDIR="$PROJECT/package_cache/tmp_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
cd "$PROJECT"
for attempt in 1 2 3 4 5 6; do
if sha256sum --check <<'SOURCE_SHA'
8885402965f156cad4453850c599d7c745379bff00d006a1378323a82f52d60a  /dfs/share/maleiLab/wwt/eeg/code/PKU_build_native_T_latents_GPU_20261109_v256.py
7710845c6c8226f9fb5b8c07d89e31688c385419b294198d885ff43d2b6f8de3  /dfs/share/maleiLab/wwt/eeg/code/native_T_latents_policy_v256.json
SOURCE_SHA
then break; fi
test "$attempt" != 6
sleep 2
done
"$PROJECT/envs/torch230_cu121_v235/bin/python" -B "$PROJECT/code/PKU_build_native_T_latents_GPU_20261109_v256.py"
