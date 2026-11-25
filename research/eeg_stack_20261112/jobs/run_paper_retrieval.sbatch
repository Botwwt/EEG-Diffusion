#!/bin/bash
#SBATCH --job-name=paper-embedding
#SBATCH --partition=GPUA40S
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
set -euo pipefail
source "$SLURM_SUBMIT_DIR/environment"
track=$1
arm=$2
sub=$SLURM_ARRAY_TASK_ID
case "$track" in
    personal|native-full) fit=$(printf '%s-%02d' "$track" "$sub") ;;
    shared) fit=shared ;;
    *) exit 2 ;;
esac
export EEG_RUNS_ROOT="$RAD_ROOT/fits/$fit"
if [[ "$track" == "native-full" ]]; then
    budgets=(official-r80)
else
    budgets=(official-r4 official-r4-balanced official-r80 fresh-r4)
fi
"$py" code/paper_retrieval.py --subject "$sub" --arm "$arm" --budgets "${budgets[@]}"
