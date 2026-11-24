#!/bin/bash
#SBATCH --job-name=h14bank
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-1
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:-$PWD}"
panels=(registered replicate)
panel="${1:-${panels[${SLURM_ARRAY_TASK_ID:-0}]}}"
case "$panel" in registered|replicate) ;; *) exit 2 ;; esac
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$ARCHIVE/package_cache/hf"
export PYTHONPATH="$PWD/h14_bank:$PWD/transfer_code:$PWD"
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
"$ARCHIVE/envs/torch230_cu121_v235/bin/python" h14_bank/render_within_photo_bank.py render \
  --root h14_bank --panel "$panel" --archive "$ARCHIVE" \
  --adapter official/ip-adapter_sdxl_vit-h.safetensors
tar -cf "h14_bank/bank-$panel.tar.partial" -C h14_bank "renders/$panel"
mv "h14_bank/bank-$panel.tar.partial" "h14_bank/bank-$panel.tar"
