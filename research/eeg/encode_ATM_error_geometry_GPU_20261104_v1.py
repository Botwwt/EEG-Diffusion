"""Reuse all old parent/O features and encode only new bypass images."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import argparse,fcntl,subprocess,time
from pathlib import Path
import numpy as np
import atm_error_geometry_contract_20261104_v1 as K
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--stage',type=int,choices=(80,400),required=True);a=ap.parse_args()
q=K.read(a.policy);out=Path(q['output_root']);parent,ctx,e,C=K.setup(q);before=K.resources(q,parent,ctx,e,C,starting=True);started=time.monotonic()
prepared=K.read(out/'prepared.json');audit=K.read(out/f'images{a.stage}_audit.json');records=K.read(K.checked(audit['ordered_records']))
repo=Path(ctx['c']['base']['ctx']['relative']['repository_root']);job=K.read(repo/'audit'/(q['job_prefix']+f'_features{a.stage}_job.json'));assert job['pid']==os.getpid() and job['status']=='RUNNING'
lock=(repo/'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==parent['GPU_UUID']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
import torch
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(parent['maximum_GPU_bytes']/torch.cuda.get_device_properties(0).total_memory)
import image_evaluation as E
model=E.dino_model();baseline=np.load(K.checked(prepared['baseline_features']),allow_pickle=False);assert baseline.shape==(400,4,768)
features=np.empty((a.stage,len(K.ARMS),768),np.float32);features[:,:4]=baseline[:a.stage,:4]
first=next(v for v in records if v['arm']=='parent' and v['query_index']==0)
replay=E.dino_features([K.checked(first['image'])],model,batch_size=1,real_images=False).float().numpy()[0]
error=float(abs(replay-baseline[0,0]).max());assert error<3e-6
begin=0
if a.stage==400:
    prefix=np.load(K.checked(K.read(out/'features80_completion.json')['features']),allow_pickle=False);features[:80]=prefix;begin=80
new=[v for v in records if v['arm'] in K.NEW_ARMS and v['query_index']>=begin]
assert len(new)==(a.stage-begin)*len(K.NEW_ARMS)
for lo in range(0,len(new),32):
    assert time.monotonic()-started<850;K.resources(q,parent,ctx,e,C)
    rows=new[lo:lo+32];values=E.dino_features([K.checked(v['image']) for v in rows],model,batch_size=32,real_images=False).float().numpy()
    for row,value in zip(rows,values):features[row['query_index'],K.ARMS.index(row['arm'])]=value
assert np.isfinite(features).all() and np.allclose(np.linalg.norm(features,axis=2),1,atol=3e-5)
assert np.array_equal(features[:,:4],baseline[:a.stage,:4])
if a.stage==400:assert np.array_equal(features[:80],prefix)
K.save_array(out/f'features{a.stage}.npy',features)
K.write(out/f'features{a.stage}_completion.json',dict(status='COMPLETE_FROZEN_NINE_ARM_DINO_FEATURES',stage=a.stage,features=K.ref(out/f'features{a.stage}.npy'),
    image_audit=K.ref(out/f'images{a.stage}_audit.json'),source=K.ref(__file__),weight_sha256=E.DINO_SHA256,
    new_images_encoded=len(new),old_baseline_query_features_reused=a.stage*4,baseline_features=prepared['baseline_features'],preserved_prefix_rows=begin,
    native_DINO_single_replay_maximum_error=error,seconds=time.monotonic()-started,resources_before=before,resources_after=K.resources(q,parent,ctx,e,C)))
