#!/bin/bash
#SBATCH --partition=GPUA40S
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-3%2
#SBATCH --time=00:15:00
set -euo pipefail
umask 077
src=${SLURM_SUBMIT_DIR:?}
: "${ARCHIVE:?}" "${BUNDLE_SHA:?}"
panels=(mix13 mix14 mix23 mix24)
panel=${panels[$SLURM_ARRAY_TASK_ID]}
work=$(mktemp -d "${TMPDIR:-/tmp}/eeg_prompt_${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf -- "$work"' EXIT
cd "$work"
for attempt in 1 2 3; do
    cp "$src/background-prompts.tar" input.tar
    if echo "$BUNDLE_SHA  input.tar" | sha256sum -c -; then break; fi
    if [ "$attempt" = 3 ]; then exit 1; fi
    sleep 3
done
tar xf input.tar
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$ARCHIVE/package_cache/hf"
export PYTHONPATH="$work/code" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
py="$ARCHIVE/envs/torch230_cu121_v235/bin/python"
args=(--root inputs --panel "$panel" --archive "$ARCHIVE" --dictionary "$src/official/dictionary.csv")
"$py" code/render_background_prompt.py render "${args[@]}" --adapter "$src/official/ip-adapter_sdxl_vit-h.safetensors"
"$py" code/render_background_prompt.py name "${args[@]}"
dst="$src/background_prompt_images"
mkdir -p "$dst"
tar cf "$dst/$panel.tar.partial.$SLURM_JOB_ID" -C inputs "native_audit/renders/$panel/sub-04" "texture/renders/$panel/sub-04"
mv "$dst/$panel.tar.partial.$SLURM_JOB_ID" "$dst/$panel.tar"
sha256sum "$dst/$panel.tar"
