#!/bin/bash
#SBATCH --partition=GPUA40S
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-3%2
#SBATCH --time=00:20:00
set -euo pipefail
umask 077
src=${SLURM_SUBMIT_DIR:?}
: "${ARCHIVE:?}" "${CODE_SHA:?}"
panels=(mix13 mix14 mix23 mix24)
panel=${panels[$SLURM_ARRAY_TASK_ID]}
work=$(mktemp -d "${TMPDIR:-/tmp}/fresh_mix_${SLURM_JOB_ID}_${panel}.XXXXXX")
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
fetch fresh_mix_blind_inputs.tar 94cd4280a6885a9a1a359ee674f2cb7a73edd5db95f6930dfc33708a6c38ad76
fetch fresh_mix_image_code.tar "$CODE_SHA"
tar xf fresh_mix_blind_inputs.tar
tar xf fresh_mix_image_code.tar
echo "7fb74944ab647d9931a971c4814a09cd1650d84f890bbb1adcfc587c236a7c6d  image_protocol.json" | sha256sum -c -
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$ARCHIVE/package_cache/hf"
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONPATH="$work/code"
py="$ARCHIVE/envs/torch230_cu121_v235/bin/python"
dict="$src/official/dictionary.csv"
"$py" code/render_fresh_mix_images.py --root inputs --panel "$panel" --archive "$ARCHIVE" \
    --adapter "$src/official/ip-adapter_sdxl_vit-h.safetensors" --dictionary "$dict"
"$py" code/name_fresh_mix_images.py --root inputs --panel "$panel" --archive "$ARCHIVE" --dictionary "$dict"
dest="$src/fresh_mix_images"
mkdir -p "$dest"
tar cf "$dest/$panel.tar.partial.$SLURM_JOB_ID" -C inputs "legacy/renders/$panel/sub-04" "learned/renders/$panel/sub-04" "scores/$panel"
mv "$dest/$panel.tar.partial.$SLURM_JOB_ID" "$dest/$panel.tar"
sha256sum "$dest/$panel.tar"
