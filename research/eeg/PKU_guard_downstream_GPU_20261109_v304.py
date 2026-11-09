"""Resource and real CUDA RNG roundtrip guard around unchanged registered numerical sources."""
import json,os,shutil,socket,subprocess,sys,time
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
import torch
R=Path('/dfs/share/maleiLab/wwt/eeg');assert torch.cuda.is_available() and torch.cuda.device_count()==1
assert shutil.disk_usage(R).free>100<<30
# Validate the real installed RNG representation by restoring and replaying draws, not by guessed length.
g=torch.Generator(device='cuda').manual_seed(1729);state=g.get_state();first=torch.randn((16,),device='cuda',generator=g);g.set_state(state);second=torch.randn((16,),device='cuda',generator=g);assert torch.equal(first,second)
del first,second;torch.cuda.empty_cache()
mode=sys.argv[1];rest=sys.argv[2:];names={'train':'PKU_queue_downstream_GPU_20261109_v303.py','images':'PKU_generate_downstream_images_GPU_20261109_v303.py','score':'PKU_score_downstream_images_GPU_20261109_v303.py'};assert mode in names
subprocess.check_call([str(R/'envs/torch230_cu121_v235/bin/python'),'-B',str(R/'code'/names[mode])]+rest)
caps={'EEG_downstream_v303':32<<30,'EEG_downstream_images_v303':6<<30,'EEG_downstream_scores_v303':2<<30};sizes={}
for folder,cap in caps.items():
    n=sum(p.stat().st_size for p in (R/'runs'/folder).rglob('*') if p.is_file());assert n<cap,(folder,n,cap);sizes[folder]=n
out=R/'runs/EEG_downstream_resource_v304';out.mkdir(exist_ok=True);p=out/(os.environ['SLURM_JOB_ID']+'.json');assert not p.exists();p.write_text(json.dumps(dict(status='PASS_REAL_CUDA_RNG_REPLAY_AND_PHASE_OUTPUT_CAPS',mode=mode,CUDA_RNG_bytes=state.numel(),output_bytes=sizes),indent=2)+'\n')
