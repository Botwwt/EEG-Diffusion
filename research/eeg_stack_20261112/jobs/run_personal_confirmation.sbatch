#!/bin/bash
#SBATCH --partition=GPUA40S
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-2%2
#SBATCH --time=00:20:00
set -euo pipefail
umask 077
src=$SLURM_SUBMIT_DIR
subjects=(sub-04 sub-05 sub-01)
person=${subjects[$SLURM_ARRAY_TASK_ID]}
series=${SERIES:-personal12}
output=$src/${series}_images
work=$(mktemp -d "${TMPDIR:-/tmp}/eeg_personal_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.XXXXXX")
trap 'rm -rf -- "$work"' EXIT
mkdir -p "$output"
cd "$work"
for attempt in 1 2 3; do
  cp "$src/$series-$STAGE.tar" input.tar
  if echo "$BUNDLE_SHA  input.tar" | sha256sum -c -; then break; fi
  if [ "$attempt" = 3 ]; then exit 1; fi
  sleep 4
done
tar xf input.tar
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$ARCHIVE/package_cache/hf"
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONPATH="$work/code"
py="$ARCHIVE/envs/torch230_cu121_v235/bin/python"
dict="$src/official/dictionary.csv"
case "$STAGE" in
  render)
    "$py" code/render_within_photo_ip.py --root inputs --subject "$person" --panels personal12 \
      --archive "$ARCHIVE" --dictionary "$dict" --adapter "$src/official/ip-adapter_sdxl_vit-h.safetensors"
    dest="$output/$person.tar"
    tar cf "$dest.partial" -C inputs "renders/personal12/$person"
    mv "$dest.partial" "$dest"
    ;;
  clip)
    tar xf "$output/$person.tar" -C inputs
    "$py" code/score_within_photo_ip.py --root inputs --subjects "$person" --panels personal12 \
      --reference-dir "references/$person" --archive "$ARCHIVE" --dictionary "$dict" --out "clip/$person.json"
    dest="$output/$person-clip.json"
    cp "clip/$person.json" "$dest.partial"
    mv "$dest.partial" "$dest"
    ;;
  names)
    "$py" code/score_reference_names.py --panel personal12 --references "references/$person" \
      --archive "$ARCHIVE" --dictionary "$dict" --out "$work/$person-refnames.json"
    dest="$output/$person-refnames.json"
    cp "$work/$person-refnames.json" "$dest.partial"
    mv "$dest.partial" "$dest"
    ;;
  *) exit 2 ;;
esac
