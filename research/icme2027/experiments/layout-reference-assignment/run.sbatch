#!/bin/bash
set -euo pipefail

phase="$1"
study="$2"
anchor="$SLURM_SUBMIT_DIR"
previous="$anchor/icme-assignment-layout-20261127.4UbDqc"
source <(awk '/^umask 077/{exit} {print}' "$anchor/pair-loss-r80-20261127/run-pair-loss-r80.sbatch")
umask 077
export PYTHONDONTWRITEBYTECODE=1
export EEG_RUNS_ROOT="$study/fits/shared"
checkpoint="swap_diff/draft_intra4/d2_compete4_cond_st120_s2025.pt"

case "$phase" in
  train)
    sha256sum -c "$study/source.sha256"
    source_checkpoints="$anchor/fits/shared/swap_diff"
    target="$EEG_RUNS_ROOT/swap_diff"
    mkdir -p "$target/draft_intra4"
    ln -sfn "$source_checkpoints/arm3b_rep_rawswap_condlyr_st600_s2025.pt" "$target/arm3b_rep_rawswap_condlyr_st600_s2025.pt"
    ln -sfn "$source_checkpoints/draft_intra4/z0_oof_tr14_256.pt" "$target/draft_intra4/z0_oof_tr14_256.pt"
    export PYTHONPATH="$study/code:$PYTHONPATH"
    "$py" -u "$study/code/paper_training.py" d2 --assignment-score layout-reference
    ;;
  render)
    subject="$SLURM_ARRAY_TASK_ID"
    printf -v subject_dir 'sub-%02d' "$subject"
    test -s "$EEG_RUNS_ROOT/$checkpoint"
    "$py" -u "$previous/code/paper_render.py" --subject "$subject" --arms d2 \
      --budgets dev-r4 --output-root "$study/images/layout-reference"
    "$py" -u "$RAD_ROOT/code/paper_score.py" "$study/images/layout-reference/shared/d2/dev-r4/$subject_dir" dev
    ;;
  *)
    exit 64
    ;;
esac
