#!/bin/bash
#SBATCH --partition=GPUA40S
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-7%3
#SBATCH --time=00:20:00
set -euo pipefail
umask 077
src=${SLURM_SUBMIT_DIR:?}
: "${ARCHIVE:?}" "${CODE_SHA:?}"
people=(sub-05 sub-01)
panels=(mix13 mix14 mix23 mix24)
subject=${people[$((SLURM_ARRAY_TASK_ID/4))]}
panel=${panels[$((SLURM_ARRAY_TASK_ID%4))]}
work=$(mktemp -d "${TMPDIR:-/tmp}/fresh_people_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.XXXXXX")
trap 'rm -rf -- "$work"' EXIT
cd "$work"
fetch() {
    local file=$1 expected=$2 attempt
    for attempt in 1 2 3; do
        cp "$src/$file" "$file"
        if echo "$expected  $file" | sha256sum -c -; then return; fi
        sleep 3
    done
    exit 1
}
fetch fresh_people_blind_inputs.tar d20939ac1fcf44320bdc5ea967e193726fbca81d7ade4513720cb6ac79d0b347
fetch fresh_people_image_code.tar "$CODE_SHA"
tar xf fresh_people_blind_inputs.tar
tar xf fresh_people_image_code.tar
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$ARCHIVE/package_cache/hf"
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONPATH="$work/code"
py="$ARCHIVE/envs/torch230_cu121_v235/bin/python"
args=(--root "inputs/$subject" --protocol protocol --subject "$subject" --panel "$panel" --archive "$ARCHIVE" --dictionary "$src/official/dictionary.csv")
"$py" code/render_fresh_people_images.py render "${args[@]}" --adapter "$src/official/ip-adapter_sdxl_vit-h.safetensors"
"$py" code/render_fresh_people_images.py clip "${args[@]}"
dest="$src/fresh_people_images/$subject"
mkdir -p "$dest"
tar cf "$dest/$panel.tar.partial.$SLURM_JOB_ID" -C "inputs/$subject" "legacy/renders/$panel/$subject" "learned/renders/$panel/$subject" "scores/$panel"
mv "$dest/$panel.tar.partial.$SLURM_JOB_ID" "$dest/$panel.tar"
sha256sum "$dest/$panel.tar"
