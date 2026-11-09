#!/bin/bash
#SBATCH --job-name=eeg_queue269
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
0083fe238f75ac82ef6e87acfbe54ed322e7cf7181257c647533552ba976ead6  /dfs/share/maleiLab/wwt/eeg/code/PKU_queue_EEG_candidates_GPU_20261109_v269.py
bbf8c0bd346f2d004fdc6e38810fd8ce01a29988ae35c0a8b680d4f7934f54ea  /dfs/share/maleiLab/wwt/eeg/code/PKU_train_eeg_candidates_GPU_20261109_v267.py
d1b490a05d002a7493560959e202c02ec646a3da1a5e699621e90f965f16550d  /dfs/share/maleiLab/wwt/eeg/code/PKU_eeg_candidate_core_20261109_v266.py
976abd014d3ec797f9bd5d321469586b5643883496f41ecfd2a4c1b3aeaee713  /dfs/share/maleiLab/wwt/eeg/code/PKU_audit_EEG_candidates_CPU_20261109_v268.py
93315b41af26f1191925079ecceddbdf7712371bd348b4b4edf2a780557b5270  /dfs/share/maleiLab/wwt/eeg/code/PKU_native_generator_20261109_v253.py
8a63ad3d5a7b65cf46eb02831cbaa0b1a13b9ed94fc0cedf64fd5f7731f2dee4  /dfs/share/maleiLab/wwt/eeg/code/PKU_HIVE_native_classes_20261109_v239.py
76c77ef2942c676c5e3f473ca6fca590f3888200951d671c31efc2f22c2d61b9  /dfs/share/maleiLab/wwt/eeg/code/EEG_two_core_candidates_v266.json
SOURCE_SHA
then break; fi
test "$attempt" != 6
sleep 2
done
"$PROJECT/envs/torch230_cu121_v235/bin/python" -B "$PROJECT/code/PKU_queue_EEG_candidates_GPU_20261109_v269.py" --stage "$1" --case "$2"
