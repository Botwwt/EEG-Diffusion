#!/bin/bash
#SBATCH --job-name=ipgrad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:-$PWD}"
submit_dir="$PWD"
bundle="${1:-$submit_dir/inputs.tar}"
expected="${2:-da1cd2a4789fd688e7fbfd821e73e9fbaad6add6702308e299cb0e13c371b6f4}"
[[ "$expected" =~ ^[[:xdigit:]]{64}$ ]]
stage=$(mktemp -d "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/ipgrad.XXXXXXXX")
cp -- "$bundle" "$stage/inputs.tar"
cd "$stage"
printf '%s  inputs.tar\n' "$expected" | sha256sum --check --status
tar -xf inputs.tar ip_grad/fit_ip_denoiser_gradient.py ip_grad/design.json ip_grad/inputs.json ip_grad/generic_pairs.npz
sha256sum --check --status <<'LOCK'
04b35b320138ab4c96b58949b5a835cb5ae18e9aaa441f85cd70471de7539d0a  ip_grad/fit_ip_denoiser_gradient.py
bca91b9be737b32a79a929d0ac01919b345a8391966aca9a07a56d1cb79d67aa  ip_grad/design.json
LOCK
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$ARCHIVE/package_cache/hf"
export PYTHONPATH="$stage/ip_grad${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
"$ARCHIVE/envs/torch230_cu121_v235/bin/python" ip_grad/fit_ip_denoiser_gradient.py \
  --root ip_grad --archive "$ARCHIVE" --adapter "$submit_dir/official/ip-adapter_sdxl_vit-h.safetensors" --class-batch 1
tar -cf results.tar -C ip_grad out
result_sha=$(sha256sum results.tar | cut -d' ' -f1)
mkdir -p "$submit_dir/ip_grad"
partial="$submit_dir/ip_grad/results.tar.${SLURM_JOB_ID:-local}.partial"
cp -- results.tar "$partial"
printf '%s  %s\n' "$result_sha" "$partial" | sha256sum --check --status
mv -- "$partial" "$submit_dir/ip_grad/results.tar"
printf 'Verified results.tar %s\n' "$result_sha"
