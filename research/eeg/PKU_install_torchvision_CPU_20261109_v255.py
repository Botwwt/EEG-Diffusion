"""Restore the native preprocessing library on a Slurm CPU, offline."""
import hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
R=Path('/dfs/share/maleiLab/wwt/eeg');out=R/('runs/torchvision_environment_'+os.environ['SLURM_JOB_ID']);out.mkdir();start=time.monotonic()
m=json.loads((R/'manifests/torchvision_wheel_v255.json').read_bytes());assert len(m['files'])==1;r=m['files'][0];path=R/r['relative_path']
h=hashlib.sha256()
with path.open('rb') as f:
    for b in iter(lambda:f.read(1<<20),b''):h.update(b)
assert path.stat().st_size==6999979 and h.hexdigest()=='1e516779520fc92157d6cb245ac0fd3f79872fe81c84b8593a8f2e998106f5b1'
with (out/'install.log').open('w') as log:subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',str(path)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
p=subprocess.run([sys.executable,'-m','pip','check'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=60);assert p.returncode==0,p.stdout
import torch,torchvision
from torchvision import transforms
assert torch.__version__=='2.3.0+cu121' and torchvision.__version__=='0.18.0+cu121'
preprocess=transforms.Compose([transforms.Resize(512,interpolation=transforms.InterpolationMode.BILINEAR),transforms.CenterCrop(512),transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)])
r=dict(status='PASS_NATIVE_PREPROCESSING_ENVIRONMENT_RESTORED',job_id=os.environ['SLURM_JOB_ID'],host=socket.gethostname(),seconds=time.monotonic()-start,torch=torch.__version__,torchvision=torchvision.__version__,pip_check=p.stdout,
    preprocessing=str(preprocess),new_training_or_images=0)
b=(json.dumps(r,indent=2)+'\n').encode();tmp=out/'completion.pending'
with tmp.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
assert tmp.read_bytes()==b;tmp.replace(out/'completion.json');print(json.dumps(r),flush=True)
