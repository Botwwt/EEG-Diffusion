#!/usr/bin/env bash
#SBATCH --job-name=eeg-r80-ip
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
set -euo pipefail

# From an environment containing torch, diffusers and the local Turbo assets:
# sbatch run_official_ip_adapter.sh /path/to/render_official_ip_adapter.py \
#   --archive /archive --adapter /weights/ip-adapter_sdxl_vit-h.safetensors \
#   --conditions /inputs/ip_adapter_conditions.npz --prior /inputs/h14_prediction.npy \
#   --out /runs/official-ip --batch-size 8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "${PYTHON_BIN:-python}" -u "$@"
