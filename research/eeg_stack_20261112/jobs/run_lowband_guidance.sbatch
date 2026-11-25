#!/bin/bash
#SBATCH --partition=GPUA40S
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=lowband-guidance-%j.log
set -euo pipefail
umask 077
: "${ARCHIVE:?Archive root required}" "${BUNDLE_SHA:?Input archive hash required}"
src=$SLURM_SUBMIT_DIR
work=$(mktemp -d "${TMPDIR:-/tmp}/lowband_guidance_${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf -- "$work"' EXIT
cd "$work"
for attempt in 1 2 3; do
  cp "$src/sub-05.tar" bundle.tar
  if echo "$BUNDLE_SHA  bundle.tar" | sha256sum -c -; then break; fi
  if [ "$attempt" = 3 ]; then exit 1; fi
  sleep 4
done
tar xf bundle.tar
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
py="$ARCHIVE/envs/torch230_cu121_v235/bin/python"
dictionary=${DICTIONARY:-"$src/../official/dictionary.csv"}
adapter=${ADAPTER:-"$src/../official/ip-adapter_sdxl_vit-h.safetensors"}
"$py" code/lowband_guidance.py render --root inputs --archive "$ARCHIVE" --dictionary "$dictionary" --adapter "$adapter"
"$py" code/lowband_guidance.py clip --root inputs --archive "$ARCHIVE" --dictionary "$dictionary"
tar cf "$src/sub-05-out.tar.partial" -C inputs/sub-05 renders
mv "$src/sub-05-out.tar.partial" "$src/sub-05-out.tar"
sha256sum "$src/sub-05-out.tar"
