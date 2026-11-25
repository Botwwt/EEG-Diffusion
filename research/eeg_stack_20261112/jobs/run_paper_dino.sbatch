#!/bin/bash
#SBATCH --job-name=paper-dino
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=C64M256G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:10:00
set -euo pipefail
source "$SLURM_SUBMIT_DIR/environment"
sub=$(printf 'sub-%02d' "$SLURM_ARRAY_TASK_ID")
folders=()
case "$1" in
    anchors)
        folders+=("$RAD_ROOT/images/native-full/native-paper/official-r80/$sub")
        folders+=("$RAD_ROOT/external/atm-sub$(printf '%02d' "$SLURM_ARRAY_TASK_ID")")
        ;;
    native)
        for b in official-r4 fresh-r4; do folders+=("$RAD_ROOT/images/personal/native-paper/$b/$sub"); done
        ;;
    personal|shared)
        for b in official-r4 fresh-r4 official-r80; do folders+=("$RAD_ROOT/images/$1/d2/$b/$sub"); done
        for b in official-r4 fresh-r4; do folders+=("$RAD_ROOT/images/$1/wrong/$b/$sub"); done
        ;;
    *) exit 2 ;;
esac
"$py" code/paper_dino.py "${folders[@]}" --model "$RAD_ROOT/models/paper-dinov2-small" --device cpu --cpu-threads "$SLURM_CPUS_PER_TASK"
