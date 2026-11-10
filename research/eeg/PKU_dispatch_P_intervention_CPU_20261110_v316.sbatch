#!/bin/bash
#SBATCH --job-name=eeg_P_follow316
#SBATCH --partition=C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15000M
#SBATCH --time=00:20:00
set -euo pipefail
umask 077
PROJECT=/dfs/share/maleiLab/wwt/eeg
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=""
cd "$PROJECT"
sha256sum --check <<'SOURCE_SHA'
383fc24306f735ca8f76fcd1302cd90f428d8c3d20fea896004ac89b97f7bdc9  /dfs/share/maleiLab/wwt/eeg/code/PKU_prepare_P_intervention_GPU_20261110_v316.py
ae3127e7434622d7b243b603e680eed31bf04b15cb3bb23363a5134ee88a3b7b  /dfs/share/maleiLab/wwt/eeg/code/PKU_audit_P_intervention_CPU_20261110_v316.py
31d55a73e8ff47c77f01faadc96db9dab6bfffd96a98ad5b186f64ba9cbacd75  /dfs/share/maleiLab/wwt/eeg/code/PKU_generate_P_intervention_GPU_20261110_v316.py
46477aefd1cf35ef5c289ba58ba6a515844703e705501fa8aa12dbf16b346e8e  /dfs/share/maleiLab/wwt/eeg/code/PKU_prepare_P_intervention_scoring_CPU_20261110_v316.py
6a546e6465250c6c905f46dcc32e9b4e7e692b521051b8595a8809bcb89e097d  /dfs/share/maleiLab/wwt/eeg/code/PKU_score_P_intervention_GPU_20261110_v316.py
514dea774d979ed1dc487824aca04b01455df8410b9e62f2cb56a8ce398b8026  /dfs/share/maleiLab/wwt/eeg/code/PKU_dispatch_P_intervention_CPU_20261110_v316.py
aa14778f890de55e58fd13923a06414d6f23f7361e5132dd3e3c5e0955891468  /dfs/share/maleiLab/wwt/eeg/code/PKU_analyze_P_intervention_CPU_20261110_v316.py
603f1e38adf035d2609145d9ac7931518654af3afb451d3621f5f9371486c27a  /dfs/share/maleiLab/wwt/eeg/code/EEG_P_intervention_policy_v316.json
7b912614b6c23ad1525b18da77fd888f0bb626f9b9a1c5bcac6391463a14edca  /dfs/share/maleiLab/wwt/eeg/code/EEG_P_intervention_scoring_policy_v316.json
SOURCE_SHA
"$PROJECT/envs/torch230_cu121_v235/bin/python" -B "$PROJECT/code/PKU_dispatch_P_intervention_CPU_20261110_v316.py" --hop "$1"
