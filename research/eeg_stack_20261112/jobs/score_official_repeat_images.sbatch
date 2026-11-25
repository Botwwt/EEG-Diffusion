#!/bin/bash
#SBATCH --partition=GPUA40S
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
set -euo pipefail
umask 077
src=$SLURM_SUBMIT_DIR
series=${SERIES:-repeat}
loc=${TMPDIR:-/tmp}/official_repeat_score_${SLURM_JOB_ID}
mkdir -p "$loc/render"
cd "$loc"
for attempt in 1 2 3; do
  cp "$src/$series-score-input.tar" inputs.tar
  if echo "$BUNDLE_SHA  inputs.tar" | sha256sum -c -; then break; fi
  if [ "$attempt" = 3 ]; then exit 1; fi
  sleep 4
done
tar xf inputs.tar
for arm in r4 r80 wrong; do tar xf "$src/${series}_images/$arm.tar" -C render; done
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$ARCHIVE/package_cache/hf"
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONPATH="$loc/code:$src/transfer_code:$src/free:$src/rescore:$src"
py="$ARCHIVE/envs/torch230_cu121_v235/bin/python"
"$py" code/render_official_repeat_images.py join --packet packet.npz --design design.json \
  --dictionary "$src/official/dictionary.csv" --out render
"$py" code/score_official_repeat_jpegs.py --render render --references references --archive "$ARCHIVE" \
  --dictionary "$src/official/dictionary.csv" --design design.json --out clip/summary.json
tar cf "$src/${series}_images/clip.partial" clip render/settings.json
mv "$src/${series}_images/clip.partial" "$src/${series}_images/clip.tar"
