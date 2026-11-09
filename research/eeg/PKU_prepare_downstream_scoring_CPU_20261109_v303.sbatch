#!/bin/bash
#SBATCH --job-name=eeg_down_audit303
#SBATCH --partition=C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15000M
#SBATCH --time=00:15:00
set -euo pipefail
umask 077
PROJECT=/dfs/share/maleiLab/wwt/eeg
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$PROJECT"
for attempt in 1 2 3 4 5 6; do
if sha256sum --check <<'SOURCE_SHA'
379fa96e9b90cd2b19f584b149b1c9e98e7dc96039fd6c6babe41887a03d7d6b  /dfs/share/maleiLab/wwt/eeg/code/PKU_generate_downstream_images_GPU_20261109_v303.py
8dc1b76afb84c14e974455bcaa952d1b0ce932365e1650b4bcfede16932462a8  /dfs/share/maleiLab/wwt/eeg/code/PKU_prepare_downstream_scoring_CPU_20261109_v303.py
c752465787a3bbb709d4bf3fcad5057b933dc76468c188249222f7a94fd18433  /dfs/share/maleiLab/wwt/eeg/code/PKU_score_downstream_images_GPU_20261109_v303.py
684ac8854b05dcd133a638fe49fb2b8eb82d5d0ff722ba05cad3dacdd6465205  /dfs/share/maleiLab/wwt/eeg/code/PKU_analyze_downstream_scores_CPU_20261109_v303.py
a33858c60b7dbcdae7a4ec20ba9050c9c44193d9b9bf7c67498e15a743ac83f2  /dfs/share/maleiLab/wwt/eeg/code/PKU_dispatch_downstream_scoring_CPU_20261109_v303.py
8d1316b17c05ef9411bae8d8ce3febb2cb6631ae20f3bc92ff051c54b573cd99  /dfs/share/maleiLab/wwt/eeg/code/PKU_downstream_core_20261109_v303.py
ef1933d13f52b4971ffec655c973f01ba15bed27f647dcde7cb7dfb64b8b3766  /dfs/share/maleiLab/wwt/eeg/code/PKU_train_downstream_GPU_20261109_v303.py
e4d0a48118e048993df24c497e430c61ca132470c281028d2ff1c06b275b2d06  /dfs/share/maleiLab/wwt/eeg/code/PKU_audit_downstream_CPU_20261109_v303.py
ad700b1a7b51a24b0e48a19a712ef251db23729fa67c892fe8597de72d842882  /dfs/share/maleiLab/wwt/eeg/code/PKU_queue_downstream_GPU_20261109_v303.py
e9fbda70a4758f75310aaf2f360557063036cbf3ccc16b23f2e8fb5f520e216c  /dfs/share/maleiLab/wwt/eeg/code/PKU_dispatch_downstream_start_CPU_20261109_v303.py
16587c5505c87b5160def99212c12b9ff7606fe3a4f410dcf5e0410b7c174c34  /dfs/share/maleiLab/wwt/eeg/code/PKU_prepare_downstream_images_CPU_20261109_v303.py
c646df4216292526f7d018a3d537c56b1c161119fc095b09f82393818691f7c2  /dfs/share/maleiLab/wwt/eeg/code/PKU_probe_downstream_CPU_20261109_v303.py
070aba77ac0dc7a835ba2da8277c2572e8a060f64fa5fcc59b55458ead6837e6  /dfs/share/maleiLab/wwt/eeg/code/EEG_downstream_pins_v303.json
SOURCE_SHA
then break; fi
test "$attempt" != 6
sleep 2
done
"$PROJECT/envs/torch230_cu121_v235/bin/python" -B "$PROJECT/code/PKU_prepare_downstream_scoring_CPU_20261109_v303.py" 
