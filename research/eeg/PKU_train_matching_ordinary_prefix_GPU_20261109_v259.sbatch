#!/bin/bash
#SBATCH --job-name=eeg_O_prefix259
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
130752ed3fb98cb586bd62ebc52fb0b4b402a989a1e798627f820cea87ff92f0  /dfs/share/maleiLab/wwt/eeg/code/PKU_train_matching_ordinary_GPU_20261109_v259.py
93315b41af26f1191925079ecceddbdf7712371bd348b4b4edf2a780557b5270  /dfs/share/maleiLab/wwt/eeg/code/PKU_native_generator_20261109_v253.py
2b93c0644bde268ae876f0e2b722106852b25910fedc1cffc97820e96cadf5d7  /dfs/share/maleiLab/wwt/eeg/code/PKU_ordinary_component_source_v253.py
879fb228d81d74b44f249cfa2c6e37e13ecad00962a4a2b77dd6ee2864a20c73  /dfs/share/maleiLab/wwt/eeg/code/matching_ordinary_policy_v259.json
SOURCE_SHA
then break; fi
test "$attempt" != 6
sleep 2
done
"$PROJECT/envs/torch230_cu121_v235/bin/python" -B "$PROJECT/code/PKU_train_matching_ordinary_GPU_20261109_v259.py" --stop-after-updates 4
