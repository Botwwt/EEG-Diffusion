"""Install only separate pinned evaluation dependencies, then verify all exact metric assets."""
import hashlib,json,os,runpy,shutil,socket,subprocess,sys,time,traceback
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower() and not os.environ.get('CUDA_VISIBLE_DEVICES')
R=Path('/dfs/share/maleiLab/wwt/eeg');out=R/'runs/evaluation_environment_v275';out.mkdir();start=time.monotonic()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def write(name,r):
    p=out/name;t=p.with_name(p.name+'.pending');b=(json.dumps(r,indent=2)+'\n').encode()
    with t.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    assert t.read_bytes()==b;t.replace(p)
try:
    upload=json.loads((R/'manifests/evaluation_payloads_v275.json').read_text());assert upload['status']=='ALL6_WEIGHTS_AND_5_WHEELS_STAGED_COMPUTE_SHA_PENDING'
    for r in upload['files']:
        p=Path(r['remote']);assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
    target=R/'package_cache/evaluation_runtime_v275';assert not target.exists()
    wheels=[v['remote'] for v in upload['files'] if v['relative_path'].startswith('package_cache/')]
    assert len(wheels)==5
    with (out/'pip.log').open('x') as log:subprocess.check_call([sys.executable,'-m','pip','install','--no-index','--no-deps','--no-compile','--target',str(target)]+wheels,stdout=log,stderr=subprocess.STDOUT)
    sys.path.insert(0,str(target))
    import numpy as np
    import torch,torchvision,scipy,skimage
    from skimage.metrics import structural_similarity
    from skimage.color import rgb2gray
    torch.set_num_threads(4);assert torch.__version__=='2.3.0+cu121' and torchvision.__version__=='0.18.0+cu121' and np.__version__=='1.24.3' and skimage.__version__=='0.24.0'
    assert not torch.cuda.is_initialized();assert Path(skimage.__file__).is_relative_to(target)
    rng=np.random.default_rng(1729);x=rng.random((32,32),dtype=np.float32);y=.7*x+.3*np.roll(x,3,axis=0)
    options=dict(gaussian_weights=True,sigma=1.5,use_sample_covariance=False,data_range=1.)
    original=structural_similarity(x,y,multichannel=True,**options);explicit=structural_similarity(x,y,channel_axis=None,**options);assert original==explicit
    assert structural_similarity(x,x,multichannel=True,**options)==1.
    assets=R/'assets/evaluation_v272';previous=assets/'download_status.json';assert previous.exists() and not (assets/'completion.json').exists()
    shutil.copyfile(previous,out/'original_failed_download_status.json')
    runpy.run_path(str(R/'code/PKU_restore_metric_assets_CPU_20261109_v272.py'),run_name='__main__')
    result=json.loads((assets/'completion.json').read_text());assert result['status']=='PASS_ALL6_EXACT_HISTORICAL_WEIGHT_SHA_AND_CPU_STRICT_MODELS'
    write('completion.json',dict(status='PASS_OFFLINE_EVALUATION_ENV_ALL6_MODELS_AND_SSIM',job_id=os.environ['SLURM_JOB_ID'],seconds=time.monotonic()-start,asset_completion_sha256=sha(assets/'completion.json'),
        evaluation_runtime=str(target),Torch=torch.__version__,torchvision=torchvision.__version__,NumPy=np.__version__,scipy=scipy.__version__,skimage=skimage.__version__,SSIM_source_keyword=float(original),SSIM_explicit2D=float(explicit),
        separate_target_main_training_env_unchanged=True,files_SHA_verified=len(upload['files']),GPU_used=False,SEED_caption_and_detector_still_missing=True,source_sha256=sha(__file__)))
except BaseException as e:
    write('failure.json',dict(status='FAILED_PRESERVED',error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-start));raise
