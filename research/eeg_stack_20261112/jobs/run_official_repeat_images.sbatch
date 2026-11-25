#!/bin/bash
#SBATCH --partition=GPUA40S
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-2%2
#SBATCH --time=00:18:00
set -euo pipefail
umask 077
src=$SLURM_SUBMIT_DIR
series=${SERIES:-repeat}
loc=${TMPDIR:-/tmp}/official_repeat_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$loc" "$src/${series}_images"
cd "$loc"
for attempt in 1 2 3; do
  cp "$src/$series-input.tar" input.tar
  if echo "$BUNDLE_SHA  input.tar" | sha256sum -c -; then break; fi
  if [ "$attempt" = 3 ]; then exit 1; fi
  sleep 4
done
tar xf input.tar
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$ARCHIVE/package_cache/hf"
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONPATH="$loc/code"
py="$ARCHIVE/envs/torch230_cu121_v235/bin/python"
arms=(r4 r80 wrong)
arm=${arms[$SLURM_ARRAY_TASK_ID]}
selected=("$arm" "${arm}_visual")
if [[ $series == global_angle ]]; then selected=("${arm}_visual"); fi
if [[ $series == global_named || $series == dual_r80 ]]; then selected=("$arm"); fi
"$py" code/render_official_repeat_images.py render --archive "$ARCHIVE" \
  --adapter "$src/official/ip-adapter_sdxl_vit-h.safetensors" --dictionary "$src/official/dictionary.csv" \
  --design design.json --packet packet.npz --out render --arms "${selected[@]}"
members=()
for name in "${selected[@]}"; do members+=("$name" "$name.json"); done
target="$src/${series}_images/$arm.tar"
tar cf "$target.partial" -C render "${members[@]}"
mv "$target.partial" "$target"
