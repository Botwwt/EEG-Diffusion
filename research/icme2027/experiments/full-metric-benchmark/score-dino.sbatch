#!/bin/bash
set -euo pipefail

study="$1"
anchor="$SLURM_SUBMIT_DIR"
source <(awk '/^umask 077/{exit} {print}' "$anchor/pair-loss-r80-20261127/run-pair-loss-r80.sbatch")
umask 077
export PYTHONDONTWRITEBYTECODE=1
sha256sum -c "$study/source.sha256"

work="$(mktemp -d /tmp/eeg-common-dino.XXXXXXXX)"
trap 'rm -rf -- "$work"' EXIT
printf -v subject 'sub-%02d' "$SLURM_ARRAY_TASK_ID"
methods=(brain-hive atm nice eegnet ridge m1a-d2)
budgets=(official-r4 official-r80)
archive="$anchor/uniform_reconstruction_v1/exports/render"
folders=()

for budget in "${budgets[@]}"; do
  raw="$archive/$budget/$subject.rgb.tar"
  scores="$archive/$budget/$subject.scores.tar"
  test -s "$raw"
  test -s "$scores"
  tar -xf "$raw" -C "$work"
  tar -xf "$scores" -C "$work"
  for method in "${methods[@]}"; do
    folders+=("$work/images/$method/$budget/$subject")
  done
done

"$py" -u "$study/code/paper_dino.py" "${folders[@]}" \
  --panel official --device cuda --model "$anchor/models/paper-dinov2-small" \
  --archive "$ARCHIVE" --prep "$anchor/data/prep" --cpu-threads 4

for budget in "${budgets[@]}"; do
  for method in "${methods[@]}"; do
    dest="$study/results/$budget/$method/$subject"
    mkdir -p "$dest"
    src="$work/images/$method/$budget/$subject"
    for name in dino.json dino.npz; do
      test -s "$src/$name"
      cp -- "$src/$name" "$dest/$name.partial"
      mv -- "$dest/$name.partial" "$dest/$name"
    done
  done
done
printf 'Verified independent original-image scoring for %s on %s unchanged RGB queries.\n' "$subject" 2400
