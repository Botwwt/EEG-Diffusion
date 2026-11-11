"""Private optional-package installation; existing training env and v320 remain unchanged."""
import hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
R=Path('/dfs/share/maleiLab/wwt/eeg');r=dict(path=str(R/'package_cache/wheels/einops-0.8.1-py3-none-any.whl'),sha256='919387eb55330f5757c6bea9165c5ff5cfe63a642682ea788a6d472576d81737');wheel=Path(r['path']);assert hashlib.sha256(wheel.read_bytes()).hexdigest()==r['sha256']
target=R/'package_cache/backbone_runtime_v323';assert not target.exists();start=time.monotonic()
subprocess.check_call([sys.executable,'-m','pip','install','--no-index','--no-deps','--disable-pip-version-check','--target',str(target),str(wheel)])
sys.path.insert(0,str(target));import einops
assert einops.__version__=='0.8.1' and str(einops.__file__).startswith(str(target))
r=dict(status='PASS_PRIVATE_EINOPS081_RUNTIME_ONLY',package='einops',version=einops.__version__,target=str(target),wheel_sha256=r['sha256'],seconds=time.monotonic()-start,CPU_job=os.environ['SLURM_JOB_ID'],existing_training_environment_modified=False,v320_modified=False)
(R/'manifests/EEG_backbone_private_runtime_v323.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
