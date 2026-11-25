#!/bin/bash
#SBATCH --partition=GPUA40S
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
set -euo pipefail
umask 077
src=${SLURM_SUBMIT_DIR:?}
: "${ARCHIVE:?}" "${BUNDLE_SHA:?}"
work=$(mktemp -d "${TMPDIR:-/tmp}/eeg_neighbor_${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf -- "$work"' EXIT
cd "$work"
for attempt in 1 2 3; do
    cp "$src/canonical-neighbor-personal12.tar" input.tar
    if echo "$BUNDLE_SHA  input.tar" | sha256sum -c -; then break; fi
    if [ "$attempt" = 3 ]; then exit 1; fi
    sleep 4
done
tar xf input.tar
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$ARCHIVE/package_cache/hf"
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONPATH="$work/code"
py="$ARCHIVE/envs/torch230_cu121_v235/bin/python"
"$py" code/render_canonical_neighbor_personal12.py --root inputs --archive "$ARCHIVE" \
    --adapter "$src/official/ip-adapter_sdxl_vit-h.safetensors" --dictionary "$src/official/dictionary.csv"
dest="$src/canonical_neighbor_personal12_images"
if [ -n "${SCORE_BUNDLE_SHA:-}" ]; then
    for attempt in 1 2 3; do
        cp "$src/canonical-neighbor-naming.tar" naming.tar
        if echo "$SCORE_BUNDLE_SHA  naming.tar" | sha256sum -c -; then break; fi
        if [ "$attempt" = 3 ]; then exit 1; fi
        sleep 4
    done
    tar xf naming.tar
    "$py" code/score_canonical_neighbor_images.py clip --root inputs --archive "$ARCHIVE" \
        --dictionary "$src/official/dictionary.csv"
fi
mkdir -p "$dest"
tar -cf "$dest/sub-04.tar.partial.$SLURM_JOB_ID" -C inputs renders/personal12/sub-04
mv "$dest/sub-04.tar.partial.$SLURM_JOB_ID" "$dest/sub-04.tar"
sha256sum "$dest/sub-04.tar"
