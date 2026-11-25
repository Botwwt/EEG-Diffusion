#!/bin/bash
#SBATCH --partition=GPUA40S
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
set -euo pipefail
umask 077
: "${ARCHIVE:?}" "${EEG_SHA:?}" "${ORACLE_SHA:?}"
src=${SLURM_SUBMIT_DIR:?}
work=$(mktemp -d "${TMPDIR:-/tmp}/eeg_global_${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf -- "$work"' EXIT
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$ARCHIVE/package_cache/hf"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
py="$ARCHIVE/envs/torch230_cu121_v235/bin/python"
dest="$src/global_photo_images"
mkdir -p "$dest"
for stage in eeg oracle; do
    mkdir "$work/$stage"
    cd "$work/$stage"
    hash=$EEG_SHA
    if [ "$stage" = oracle ]; then hash=$ORACLE_SHA; fi
    for attempt in 1 2 3; do
        cp "$src/global-photo-$stage.tar" input.tar
        if echo "$hash  input.tar" | sha256sum -c -; then break; fi
        if [ "$attempt" = 3 ]; then exit 1; fi
        sleep 3
    done
    tar xf input.tar
    export PYTHONPATH="$work/$stage/code"
    args=(--stage "$stage" --root inputs --archive "$ARCHIVE" --dictionary "$src/official/dictionary.csv")
    "$py" code/render_global_photo.py render "${args[@]}" --adapter "$src/official/ip-adapter_sdxl_vit-h.safetensors"
    "$py" code/render_global_photo.py name "${args[@]}"
    tar cf "$dest/$stage.tar.partial.$SLURM_JOB_ID" -C inputs renders/personal12/sub-04
    mv "$dest/$stage.tar.partial.$SLURM_JOB_ID" "$dest/$stage.tar"
    sha256sum "$dest/$stage.tar"
done
