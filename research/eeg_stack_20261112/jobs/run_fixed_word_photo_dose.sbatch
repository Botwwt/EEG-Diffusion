#!/bin/bash
#SBATCH --partition=GPUA40S
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-5%2
#SBATCH --time=00:15:00
set -euo pipefail
umask 077
src=$SLURM_SUBMIT_DIR
loc=${TMPDIR:-/tmp}/eeg_photo_dose_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$loc" "$src/photo_dose_images"
cd "$loc"
for attempt in 1 2 3; do
  cp "$src/photo-dose-input.tar" input.tar
  if echo "$BUNDLE_SHA  input.tar" | sha256sum -c -; then break; fi
  if [ "$attempt" = 3 ]; then exit 1; fi
  sleep 4
done
tar xf input.tar
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$ARCHIVE/package_cache/hf"
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONPATH="$loc/code"
py="$ARCHIVE/envs/torch230_cu121_v235/bin/python"
subjects=(single-1 cohort-3 cohort-9)
panels=(registered replicate)
person=${subjects[$((SLURM_ARRAY_TASK_ID%3))]}
panel=${panels[$((SLURM_ARRAY_TASK_ID/3))]}
"$py" code/render_fixed_word_photo_dose.py --root inputs --subject "$person" --panels "$panel" \
  --archive "$ARCHIVE" --adapter "$src/official/ip-adapter_sdxl_vit-h.safetensors" --dictionary "$src/official/dictionary.csv"
target="$src/photo_dose_images/$panel-$person.tar"
tar cf "$target.partial" -C inputs "renders/$panel/$person"
mv "$target.partial" "$target"
