#!/bin/bash
#SBATCH --partition=GPUA40S
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-1%2
#SBATCH --time=00:35:00
set -euo pipefail
umask 077
src=$SLURM_SUBMIT_DIR
loc=${TMPDIR:-/tmp}/eeg_word_guidance_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$loc/baseline" "$src/fixed_word_guidance_images"
cd "$loc"
panels=(registered replicate)
panel=${panels[$SLURM_ARRAY_TASK_ID]}
cp "$src/fixed-word-guidance-input.tar" input.tar
echo "$BUNDLE_SHA  input.tar" | sha256sum -c -
tar xf input.tar
tar xf "$src/photo_dose_images/$panel-cohort-9.tar" -C baseline "renders/$panel/cohort-9/settings.json"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$ARCHIVE/package_cache/hf"
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONPATH="$loc/code"
py="$ARCHIVE/envs/torch230_cu121_v235/bin/python"
"$py" code/render_fixed_word_guidance.py --root inputs --baseline baseline/renders --out guided \
  --registration fixed_word_guidance_design.json --panels "$panel" --archive "$ARCHIVE" \
  --adapter "$src/official/ip-adapter_sdxl_vit-h.safetensors" --dictionary "$src/official/dictionary.csv"
target="$src/fixed_word_guidance_images/$panel-cohort-9.tar"
tar cf "$target.$SLURM_JOB_ID.partial" -C guided "renders/$panel/cohort-9"
mv "$target.$SLURM_JOB_ID.partial" "$target"
