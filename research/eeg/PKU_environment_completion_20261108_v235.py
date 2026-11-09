"""Executed in Slurm CPU allocation, never on login."""
import json,os,socket,sys,time
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
assert 'login' not in socket.gethostname().lower()
import torch,numpy,scipy
p=Path('/dfs/share/maleiLab/wwt/eeg/manifests/environment_completion_20261108_v235.json')
assert not p.exists()
data=dict(status='COMPLETE_SLURM_CPU_PROJECT_ENVIRONMENT',job_id=os.environ['SLURM_JOB_ID'],hostname=socket.gethostname(),
          python=sys.version,torch=torch.__version__,numpy=numpy.__version__,scipy=scipy.__version__,CUDA_runtime=torch.version.cuda,
          package_data_and_environment_under_wwt=True,GPU_allocated=False)
tmp=p.with_suffix('.pending');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(p)
print(json.dumps(data))
