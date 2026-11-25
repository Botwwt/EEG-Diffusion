#!/bin/bash
#SBATCH --partition=GPUA40S
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=support-view-%j.log
set -euo pipefail
umask 077
: "${ARCHIVE:?Archive root required}" "${BUNDLE_SHA:?Input archive hash required}" "${SUBJECT:?Participant required}"
src=$SLURM_SUBMIT_DIR
work=$(mktemp -d "${TMPDIR:-/tmp}/support_view_${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf -- "$work"' EXIT
cd "$work"
for attempt in 1 2 3; do
  cp "$src/$SUBJECT.tar" bundle.tar
  if echo "$BUNDLE_SHA  bundle.tar" | sha256sum -c -; then break; fi
  if [ "$attempt" = 3 ]; then exit 1; fi
  sleep 4
done
tar xf bundle.tar
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
py="$ARCHIVE/envs/torch230_cu121_v235/bin/python"
args=(--archive "$ARCHIVE" --dictionary "$src/../official/dictionary.csv" --subject "$SUBJECT" --root inputs)
"$py" code/support_view.py render "${args[@]}" --adapter "$src/../official/ip-adapter_sdxl_vit-h.safetensors"
"$py" code/support_view.py clip "${args[@]}"
tar cf "$src/$SUBJECT-out.tar.partial" -C "inputs/$SUBJECT" renders
mv "$src/$SUBJECT-out.tar.partial" "$src/$SUBJECT-out.tar"
sha256sum "$src/$SUBJECT-out.tar"
