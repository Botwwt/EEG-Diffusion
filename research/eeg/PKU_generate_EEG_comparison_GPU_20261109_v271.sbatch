#!/bin/bash
#SBATCH --job-name=eeg_images271
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
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME="$PROJECT/package_cache/hf" XDG_CACHE_HOME="$PROJECT/package_cache/xdg"
cd "$PROJECT"
for attempt in 1 2 3 4 5 6; do
if sha256sum --check <<'SOURCE_SHA'
5ecb79fb4fdcd0a46c6a1af9c839232cc62066143c6953ebf5dfa2e789a73228  /dfs/share/maleiLab/wwt/eeg/code/PKU_prepare_EEG_image_conditions_CPU_20261109_v271.py
83887037cbf651ddccadf0afe13ec18908ba3e0c115d1c96d93a9243d6daca04  /dfs/share/maleiLab/wwt/eeg/code/PKU_generate_EEG_comparison_GPU_20261109_v271.py
12e0ce367a848fc5358f583a5ed4c250a09ea178329be2ed33cb0816221e5f4b  /dfs/share/maleiLab/wwt/eeg/code/PKU_dispatch_EEG_images_CPU_20261109_v271.py
435c56d7e97cb56326554f17e35eb286724b3d0cc653f7cc3ce3468b4c8d87ce  /dfs/share/maleiLab/wwt/eeg/code/EEG_image_comparison_v271.json
93315b41af26f1191925079ecceddbdf7712371bd348b4b4edf2a780557b5270  /dfs/share/maleiLab/wwt/eeg/code/PKU_native_generator_20261109_v253.py
8a63ad3d5a7b65cf46eb02831cbaa0b1a13b9ed94fc0cedf64fd5f7731f2dee4  /dfs/share/maleiLab/wwt/eeg/code/PKU_HIVE_native_classes_20261109_v239.py
d1b490a05d002a7493560959e202c02ec646a3da1a5e699621e90f965f16550d  /dfs/share/maleiLab/wwt/eeg/code/PKU_eeg_candidate_core_20261109_v266.py
2b93c0644bde268ae876f0e2b722106852b25910fedc1cffc97820e96cadf5d7  /dfs/share/maleiLab/wwt/eeg/code/PKU_ordinary_component_source_v253.py
SOURCE_SHA
then break; fi
test "$attempt" != 6
sleep 2
done
"$PROJECT/envs/torch230_cu121_v235/bin/python" -B "$PROJECT/code/PKU_generate_EEG_comparison_GPU_20261109_v271.py" --stage "$1"
