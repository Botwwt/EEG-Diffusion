#!/bin/bash
#SBATCH --job-name=paper-fullnative
#SBATCH --partition=GPUA40S
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
set -euo pipefail
source "$SLURM_SUBMIT_DIR/environment"
subject=$SLURM_ARRAY_TASK_ID
subject_id=$(printf 'sub-%02d' "$subject")
export EEG_RUNS_ROOT="$RAD_ROOT/fits/$(printf 'native-full-%02d' "$subject")"
checkpoint="$EEG_RUNS_ROOT/enc/native/$subject_id/fold-1_seed2025/brain_model.pt"
if [[ ! -f "$checkpoint" ]]; then
    "$py" code/train.py --sub "$subject" --tag native --paper-fit-only --fold -1
fi
"$py" code/paper_render.py --subject "$subject" --arms native-paper native-code --budgets official-r80
for arm in native-paper native-code; do
    output="$RAD_ROOT/images/native-full/$arm/official-r80/$subject_id"
    if [[ ! -f "$output/scores.json" ]]; then
        "$py" code/paper_score.py "$output" official
    fi
done
