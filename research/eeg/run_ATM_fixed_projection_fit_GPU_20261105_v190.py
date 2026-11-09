"""Matched full-T continuation under an immutable old output projection."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import argparse,copy,fcntl,hashlib,json,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import atm_fixed_projection_contract_20261105_v190 as C
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--mode',required=True,choices=['fit','conditions']);ap.add_argument('--end',type=int,default=0);a=ap.parse_args()
start=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;c=C.setup(a.policy);q,K,tc=c['q'],c['K'],c['train'];out=c['fit'];pq=tc['physical'];assert K.ref(__file__)==q['sources']['fit']
from atm_repeat_image_clean_cache_20261105_v2 import release
K.write(out/('cache_'+a.mode+str(a.end)+'.json'),release(pq['physical_source']));before=C.resources(c,True)
suffix=a.mode+('_'+str(a.end) if a.mode=='fit' else '');job=K.read(c['repo']/'audit'/(q['job_prefix']+'_'+suffix+'_job.json'));assert job['pid']==os.getpid() and job['status']=='RUNNING'
lock=(c['repo']/'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==pq['GPU_UUID']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
sys.path[:0]=pq['existing_runtime_paths']+['/root/atm_official_20261016']
import torch
from models.atms import ATMS
from atm_conditional_stimulus_objective_20261104_v1 import torch_loss
import atm_actual_NCE_Adam_repeat_core_20261105_v1 as G
import atm_fixed_projection_training_core_20261105_v190 as H
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(pq['maximum_GPU_bytes']/torch.cuda.get_device_properties(0).total_memory)
def guard():
 assert time.monotonic()-start<850 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<pq['maximum_extra_CPU_bytes']
 assert torch.cuda.max_memory_allocated()<pq['maximum_GPU_bytes']
 used=K.allocated(q['project_root']);assert used<q['maximum_output_bytes'];fs=os.statvfs(q['project_root']);assert fs.f_bavail*fs.f_frsize>=before['required_including_this_project']+before['own_used_bytes']-used
def save(p,state):
 assert not p.exists();tmp=p.with_name(p.name+'.pending');assert not tmp.exists();torch.save(state,tmp);tmp.replace(p)
def checkpoint():
 op=copy.deepcopy(opt.state_dict())
 for st in op['state'].values():
  for key,v in st.items():
   if torch.is_tensor(v):st[key]=v.cpu()
 return dict(model={n:v.detach().cpu().clone() for n,v in net.state_dict().items()},optimizer=op,epoch=epoch+1,updates=updates,exposures=exposures,arm='fixed_N_head',parent=tc['q']['checkpoint'],policy=K.ref(a.policy),CPU_rng=torch.get_rng_state(),CUDA_rng=torch.cuda.get_rng_state())
try:
 base=torch.load(K.checked(tc['q']['checkpoint']),map_location='cpu',weights_only=True,mmap=True);assert base['updates']==830 and base['epoch']==10
 mean=torch.load(K.checked(pq['EEG']),map_location='cpu',weights_only=True,mmap=True)
 if a.mode=='fit':
  assert a.end in [5,10];initial=base if a.end==5 else torch.load(out/'epoch05.pt',map_location='cpu',weights_only=True,mmap=True)
  net=ATMS().cuda();net.load_state_dict(initial['model'],strict=True);net.train();opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=.01);opt.load_state_dict(copy.deepcopy(initial['optimizer']));meta=G.F.metadata(net)
  targets=torch.load(K.checked(pq['visual_target']),map_location='cpu',weights_only=True,mmap=True);updates=0 if a.end==5 else initial['updates'];exposures=0 if a.end==5 else initial['exposures']
  oldepochs=[json.loads(s) for s in (Path(tc['q']['output_root'])/'fits/R4_continued/epochs.jsonl').read_text().splitlines()]
  for epoch in range(a.end-5,a.end):
   order=np.random.default_rng(1739+epoch).permutation(1323);coverage=np.zeros(13230,np.uint8);seen=0;loss_sum=norm_sum=0.;clipped=0
   for bi,lo in enumerate(range(0,1323,16)):
    guard();pos=(order[lo:lo+16,None]*10+np.arange(10)).reshape(-1);ids=tc['T'][pos];coverage[pos]+=1
    loss,gs,_=H.native_gradients(net,meta,mean[ids].cuda(),targets[ids].cuda(),101729+epoch*10000+bi,torch_loss)
    info=H.assign_clip_mask(net,meta,gs,True);opt.step();assert all(torch.isfinite(p).all() for p in net.parameters())
    seen+=len(ids);updates+=1;exposures+=len(ids);loss_sum+=float(loss)*len(ids);norm_sum+=info['preclip_full_parameter_norm'];clipped+=info['preclip_full_parameter_norm']>1
    if bi%16==0:K.write(out/'progress.json',dict(phase='FIT_FIXED_PROJECTION',epoch=epoch+1,batch=bi+1,updates=updates,exposures=exposures))
   assert seen==13230 and np.all(coverage==1);order_sha=hashlib.sha256(order.tobytes()).hexdigest();assert order_sha==oldepochs[epoch]['concept_order_sha256']
   for n,p in net.state_dict().items():
    if n.startswith('proj_eeg.'):assert torch.equal(p.cpu(),base['model'][n])
   event=dict(epoch=epoch+1,updates=updates,exposures=exposures,coverage_min=int(coverage.min()),coverage_max=int(coverage.max()),concept_order_sha256=order_sha,online_NCE=loss_sum/seen,mean_preclip_norm=norm_sum/83,clipped_batches=clipped,effective_logit_scale=float(net.logit_scale.exp().clamp_max(100)),projection_exact_base=True,seconds=time.monotonic()-start)
   with (out/'epochs.jsonl').open('a') as f:f.write(json.dumps(event)+'\n');f.flush();os.fsync(f.fileno())
   print(json.dumps(event),flush=True)
  assert updates==83*a.end and exposures==13230*a.end;save(out/f'epoch{a.end:02d}.pt',checkpoint())
  result=dict(status='COMPLETE_FIXED_PROJECTION_FULL_T_EPOCH_BOUNDARY',epoch=a.end,updates=updates,exposures=exposures,checkpoint=K.ref(out/f'epoch{a.end:02d}.pt'))
 else:
  done=K.read(out/'fit_10_completion.json');assert done['updates']==830;st=torch.load(K.checked(done['checkpoint']),map_location='cpu',weights_only=True,mmap=True)
  net=ATMS().cuda();net.load_state_dict(st['model'],strict=True);net.eval();test=torch.load(K.checked(tc['R80']['EEG_R80']),map_location='cpu',weights_only=True,mmap=True)
  ids=[r['array_row'] for r in tc['queries']];values=np.empty((600,1024),np.float32)
  with torch.inference_mode():
   for lo in range(0,600,100):
    guard();x=mean[ids[lo:lo+100]] if lo<400 else test[lo-400:lo-300]
    values[lo:lo+100]=net(x.cuda(),torch.ones(len(x),device='cuda',dtype=torch.long)).cpu().numpy()
  assert np.isfinite(values).all();assert not (out/'conditions.npy').exists();np.save(out/'conditions.npy',values)
  result=dict(status='COMPLETE_FIXED_HEAD_EEG_ONLY600_CONDITIONS',conditions=K.ref(out/'conditions.npy'),checkpoint=done['checkpoint'],normal_inference_only_EEG=True)
 guard();result.update(seconds=time.monotonic()-start,extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0,peak_GPU_bytes=torch.cuda.max_memory_allocated());K.write(out/(suffix+'_completion.json'),result)
except BaseException as exc:
 partial=None
 if a.mode=='fit' and all(k in globals() for k in ['net','opt','epoch','updates','exposures']):
  try:save(out/f'failed_end{a.end}.pt',checkpoint());partial=K.ref(out/f'failed_end{a.end}.pt')
  except BaseException as err:partial=dict(error=repr(err))
 K.write(out/(suffix+'_failure.json'),dict(error=repr(exc),traceback=traceback.format_exc(),partial=partial));raise
