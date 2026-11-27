#!/bin/bash
set -euo pipefail

stage="$1"
study="$2"
anchor="$SLURM_SUBMIT_DIR"
source <(awk '/^umask 077/{exit} {print}' "$anchor/pair-loss-r80-20261127/run-pair-loss-r80.sbatch")
umask 077
export PYTHONDONTWRITEBYTECODE=1

configure_arm() {
  local arm="$1"
  export EEG_RUNS_ROOT="$study/fits/$arm/shared"
  local source="$anchor/fits/shared/swap_diff"
  local target="$EEG_RUNS_ROOT/swap_diff"
  if [[ "$stage" == "train" ]]; then
    mkdir -p "$target/draft_intra4"
    ln -sfn "$source/arm3b_rep_rawswap_condlyr_st600_s2025.pt" "$target/arm3b_rep_rawswap_condlyr_st600_s2025.pt"
    ln -sfn "$source/draft_intra4/z0_oof_tr14_256.pt" "$target/draft_intra4/z0_oof_tr14_256.pt"
  else
    test -s "$target/draft_intra4/d2_compete4_cond_st120_s2025.pt"
  fi
}

case "$stage" in
  train)
    case "$SLURM_ARRAY_TASK_ID" in
      0) arm=individual ;;
      1) arm=group-mean ;;
      *) exit 64 ;;
    esac
    sha256sum -c "$study/source.sha256"
    configure_arm "$arm"
    export PYTHONPATH="$study/code:$PYTHONPATH"
    "$py" -u "$study/code/paper_training.py" d2 --training-layout "$arm"
    ;;
  render)
    subject="$SLURM_ARRAY_TASK_ID"
    printf -v subject_dir 'sub-%02d' "$subject"
    for arm in individual group-mean; do
      configure_arm "$arm"
      images="$study/images/$arm"
      "$py" -u "$study/code/paper_render.py" --subject "$subject" \
        --arms d2 --budgets dev-r4 --output-root "$images"
      "$py" -u "$RAD_ROOT/code/paper_score.py" "$images/shared/d2/dev-r4/$subject_dir" dev
    done
    ;;
  *) exit 64 ;;
esac
