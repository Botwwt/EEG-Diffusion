"""Only EEG creates the bypass conditions; no visual teacher is an inference input."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import argparse,fcntl,gc,resource,subprocess,sys,time
from pathlib import Path
import numpy as np
import atm_prior_factorial_contract_20261104_v1 as K
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);a=ap.parse_args();q=K.read(a.policy);out=Path(q['output_root']);started=time.monotonic()
parent,ctx,e,C=K.setup(q);before=K.resources(q,parent,ctx,e,C,starting=True);b=ctx['c']['base'];old=ctx['old'];prepared=ctx['old_prepared']
repo=Path(b['ctx']['relative']['repository_root']);job=K.read(repo/'audit'/(q['job_prefix']+'_prepare_job.json'))
assert job['pid']==os.getpid() and job['status']=='RUNNING'
lock=(repo/'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==parent['GPU_UUID']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
import torch
from scipy.special import erf
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(parent['maximum_GPU_bytes']/torch.cuda.get_device_properties(0).total_memory)
sys.path.insert(0,'/root/atm_official_20261016');from models.atms import ATMS
assert K.sha('/root/atm_official_20261016/models/atms.py')==b['parent']['native_encoder_source_sha256']
queries=K.read(K.checked(prepared['queries']));rows=[v['array_row'] for v in queries];assert len(rows)==len(set(rows))==400 and not set(rows)&set(b['T'])
full=next(v for v in b['parent_done']['banks'] if v['role']=='full');permissions=K.read(K.checked(full['permissions']))
assert set(permissions['training_rows'])==set(b['T']) and set(rows)<=set(permissions['export_rows'])
raw=torch.load(K.checked(b['sub']['input']),map_location='cpu',weights_only=True,mmap=True)
encoder=ATMS().eval().requires_grad_(False);encoder.load_state_dict(torch.load(K.checked(b['sub']['models']['full']['checkpoint']),map_location='cpu',weights_only=True,mmap=True),strict=True)
with torch.inference_mode():cpu=encoder(raw[rows[:8]],torch.ones(8,dtype=torch.long)).numpy()
encoder.cuda();parts=[]
with torch.inference_mode():
    for lo in range(0,400,128):
        ix=rows[lo:lo+128];parts.append(encoder(raw[ix].cuda(),torch.ones(len(ix),device='cuda',dtype=torch.long)).float().cpu().numpy())
direct=np.concatenate(parts);assert direct.shape==(400,1024) and np.isfinite(direct).all()
err=float(np.max(abs(direct[:8]-cpu)/(1+abs(cpu))));assert err<3e-5
del encoder,parts,raw;gc.collect();torch.cuda.empty_cache()
from atm_native_bounded_ordinary_20261103_v1 import NativePriorOrdinary
state=torch.load(K.checked(q['ordinary_state']),map_location='cpu',weights_only=True);assert state['updates']==2481 and state['exposures']==39690 and torch.all(state['coverage']==3)
head=NativePriorOrdinary().eval().requires_grad_(False);head.load_state_dict(state['model'],strict=True)
old_conditions=np.load(K.checked(prepared['conditions']),allow_pickle=False);assert old_conditions.shape==(400,4,1024)
values=np.empty((400,4,1024),np.float32);values[:,:2]=old_conditions[:,:2];values[:,2]=direct
w={k:v.numpy().astype(np.float64) for k,v in state['model'].items()};maximum=0.;zero=NativePriorOrdinary().eval().requires_grad_(False)
with torch.inference_mode():
    for i in range(400):
        x=torch.from_numpy(direct[i:i+1].copy());assert torch.equal(zero(x)[0],x) and torch.equal(head(x,enabled=False)[0],x)
        values[i,3]=head(x)[0].numpy()[0]
        xx=direct[i:i+1].astype(np.float64);u=xx-xx.mean(1,keepdims=True);n=np.linalg.norm(u,axis=1,keepdims=True)
        h=u*32/n;h=h@w['trunk.0.weight'].T+w['trunk.0.bias'];h=.5*h*(1+erf(h/np.sqrt(2)))
        d=h@w['direction.weight'].T+w['direction.bias'];d-=d.mean(1,keepdims=True);d-=np.sum(d*u,1,keepdims=True)/n**2*u
        delta=.25*n*d/np.sqrt(n*n+(d*d).sum(1,keepdims=True));scale=n/np.linalg.norm(u+delta,axis=1,keepdims=True)
        expected=xx+(scale-1)*u+scale*delta;maximum=max(maximum,float(abs(expected-values[i:i+1,3]).max()))
assert maximum<3e-5 and np.array_equal(values[:,:2],old_conditions[:,:2])
K.save_array(out/'conditions.npy',values);K.write(out/'queries.json',queries)
old_feature_done=K.read(Path(old['output_root'])/'features400_completion.json');K.checked(old_feature_done['features'])
K.write(out/'prepared.json',dict(status='COMPLETE_FIXED_FINAL_NATIVE_ATM_FOUR_ARM_CONDITIONS_CPU',policy=K.ref(a.policy),
    conditions=K.ref(out/'conditions.npy'),queries=K.ref(out/'queries.json'),target_feature_ref=prepared['target_feature_ref'],target_metadata_ref=prepared['target_metadata_ref'],
    baseline_features=old_feature_done['features'],ordinary_state=q['ordinary_state'],full_encoder=b['sub']['models']['full']['checkpoint'],
    direct_encoder_GPU_vs_CPU8_scaled_error=err,all400_NumPy64_head_maximum_error=maximum,all400_zero_and_disabled_conditions_byte_exact=True,
    prior_parent_and_ordinary400_condition_bytes_unchanged=True,full_encoder_fit_T_excludes_all400_Dcheck=True,ordinary_strength=1,ordinary_radius=.25,
    normal_inference_only_EEG=True,no_visual_teacher_at_bypass_input=True,new_training_updates=0,seconds=time.monotonic()-started,
    resources_before=before,resources_after=K.resources(q,parent,ctx,e,C)))
