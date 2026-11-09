"""Offline generator dependencies installed and checked only on allocated CPU."""
import hashlib,json,os,subprocess,sys,time,socket
from pathlib import Path
ROOT=Path('/dfs/share/maleiLab/wwt/eeg');assert os.environ.get('SLURM_JOB_ID')
t=time.monotonic();m=json.loads((ROOT/'manifests/native_generator_wheels_20261109_v243.json').read_text())
out=ROOT/('runs/generator_environment_'+os.environ['SLURM_JOB_ID']);out.mkdir()
for r in m['files']:
    p=ROOT/r['relative_path'];h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    assert p.stat().st_size==r['bytes'] and h.hexdigest()==r['sha256']
cmd=[sys.executable,'-m','pip','install','--no-index','--no-deps']+[str(ROOT/r['relative_path']) for r in m['files']]
with (out/'install.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
p=subprocess.run([sys.executable,'-m','pip','check'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=60)
(out/'pip_check.log').write_text(p.stdout);assert p.returncode==0,p.stdout
import torch,diffusers,transformers,accelerate,safetensors
from diffusers import StableDiffusionXLPipeline,UNet2DConditionModel,AutoencoderKL,EulerAncestralDiscreteScheduler,DDPMScheduler
r=dict(status='COMPLETE_OFFLINE_GENERATOR_ENVIRONMENT',job_id=os.environ['SLURM_JOB_ID'],host=socket.gethostname(),
    seconds=time.monotonic()-t,torch=torch.__version__,diffusers=diffusers.__version__,transformers=transformers.__version__,
    accelerate=accelerate.__version__,safetensors=safetensors.__version__,wheels=len(m['files']),pip_check=p.stdout,
    native_SDXL_pipeline_import=True,new_updates=0,new_images=0)
b=(json.dumps(r,indent=2)+'\n').encode();tmp=out/'completion.pending'
with tmp.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
assert tmp.read_bytes()==b;tmp.replace(out/'completion.json');print(json.dumps(r),flush=True)
