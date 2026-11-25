#!/bin/bash
#SBATCH --partition=GPUA40S
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
set -euo pipefail
umask 077
: "${ARCHIVE:?}" "${INPUT_SHA:?}"
src=${SLURM_SUBMIT_DIR:?}
work=$(mktemp -d "${TMPDIR:-/tmp}/eeg_protected_${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf -- "$work"' EXIT
cd "$work"
cp "$src/protected_photo.tar" input.tar
echo "$INPUT_SHA  input.tar" | sha256sum -c -
tar xf input.tar
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$ARCHIVE/package_cache/hf"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONPATH="$work/code"
py="$ARCHIVE/envs/torch230_cu121_v235/bin/python"
args=(--root inputs --archive "$ARCHIVE" --dictionary "$src/official/dictionary.csv")
"$py" code/render_protected_photo.py render "${args[@]}" --adapter "$src/official/ip-adapter_sdxl_vit-h.safetensors"
"$py" code/render_protected_photo.py name "${args[@]}"
mkdir -p "$src/protected_photo_images"
dest="$src/protected_photo_images/mix13.tar"
tar cf "$dest.partial.$SLURM_JOB_ID" -C inputs renders/mix13/sub-04
mv "$dest.partial.$SLURM_JOB_ID" "$dest"
sha256sum "$dest"
