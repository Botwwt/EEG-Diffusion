#!/bin/bash
#SBATCH --partition=GPUA40S
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-1%2
#SBATCH --time=00:12:00
set -euo pipefail
umask 077
src=$SLURM_SUBMIT_DIR
loc=${TMPDIR:-/tmp}/photo_dose_names_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$loc"
cd "$loc"
cp "$src/photo-dose-score.tar" input.tar
echo "$BUNDLE_SHA  input.tar" | sha256sum -c -
tar xf input.tar
panels=(registered replicate)
panel=${panels[$SLURM_ARRAY_TASK_ID]}
cp -r inputs guided
for s in single-1 cohort-3 cohort-9; do
  tar xf "$src/photo_dose_images/$panel-$s.tar" -C inputs
done
tar xf "$src/fixed_word_guidance_images/$panel-cohort-9.tar" -C guided
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$ARCHIVE/package_cache/hf"
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONPATH="$loc/code:$src/transfer_code:$src/free:$src/rescore:$src"
py="$ARCHIVE/envs/torch230_cu121_v235/bin/python"
args=(--fixed-word-dose --archive "$ARCHIVE" --reference-dir reference --dictionary "$src/official/dictionary.csv" --panels "$panel")
"$py" code/score_within_photo_ip.py "${args[@]}" --root inputs --subjects single-1 cohort-3 cohort-9 --out ordinary.json
"$py" code/score_within_photo_ip.py "${args[@]}" --root guided --subjects cohort-9 --out guided.json
tar cf "$src/photo_dose_images/$panel-names.partial" ordinary.json guided.json
mv "$src/photo_dose_images/$panel-names.partial" "$src/photo_dose_images/$panel-names.tar"
