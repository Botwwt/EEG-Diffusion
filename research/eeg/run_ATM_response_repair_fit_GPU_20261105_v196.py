"""Two matched full-T EEG fits; frozen visual/NN/CN teachers are training only."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import argparse,copy,fcntl,hashlib,json,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import atm_response_repair_contract_20261105_v196 as C
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--mode',choices=['fit','conditions'],required=True);ap.add_argument('--arm',default='');ap.add_argument('--end',type=int,default=0);a=ap.parse_args()
c=C.setup(a.policy);q,K,tc=c['q'],c['K'],c['train'];pq=tc['physical'];out=c['fit'];started=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
assert K.ref(__file__)==q['sources']['fit'];before=C.resources(c,True)
suffix='fit_'+a.arm+'_'+str(a.end) if a.mode=='fit' else 'conditions'
job=K.read(c['repo']/'audit'/(q['job_prefix']+'_'+suffix+'_job.json'));assert job['pid']==os.getpid() and job['status']=='RUNNING'
lock=(c['repo']/'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==pq['GPU_UUID']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
sys.path[:0]=pq['existing_runtime_paths']+['/root/atm_official_20261016']
import torch
from models.atms import ATMS
from safetensors import safe_open
import atm_response_contrast_repair_core_20261105_v195 as H
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction((20<<30)/torch.cuda.get_device_properties(0).total_memory)
def guard():
 assert time.monotonic()-started<850 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<10<<30
 assert torch.cuda.max_memory_allocated()<20<<30;used=K.allocated(q['project_root']);assert used<q['maximum_output_bytes']
 fs=os.statvfs(out);assert fs.f_bavail*fs.f_frsize>=before['required_including_this_project']+before['own_used_bytes']-used
def save(p,st):
 assert not p.exists();tmp=p.with_name(p.name+'.pending');assert not tmp.exists();torch.save(st,tmp);tmp.replace(p)
def state_record():
 op=copy.deepcopy(opt.state_dict())
 for st in op['state'].values():
  for k,v in st.items():
   if torch.is_tensor(v):st[k]=v.cpu()
 return dict(model={n:v.detach().cpu().clone() for n,v in net.state_dict().items()},optimizer=op,epoch=epoch+1,updates=updates,exposures=exposures,arm=a.arm,policy=K.ref(a.policy),CPU_rng=torch.get_rng_state(),CUDA_rng=torch.cuda.get_rng_state())
try:
 src={k:torch.load(K.checked(v),map_location='cpu',weights_only=True,mmap=True)['model'] for k,v in q['models'].items()}
 initial={n:src['NN'][n] if n.startswith('proj_eeg.') else v for n,v in src['CC'].items()}
 mean=torch.load(K.checked(pq['EEG']),map_location='cpu',weights_only=True,mmap=True)
 if a.mode=='fit':
  assert a.arm in C.ARMS[2:] and a.end in [5,10];dest=out/a.arm;dest.mkdir(exist_ok=True)
  net=ATMS().cuda();net.load_state_dict(initial,strict=True);params=H.prepare_student(net);opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01)
  updates=exposures=0
  if a.end==10:
   st=torch.load(K.checked(K.read(dest/'fit_5_completion.json')['checkpoint']),map_location='cpu',weights_only=True,mmap=True)
   assert st['epoch']==5 and st['updates']==415;net.load_state_dict(st['model'],strict=True);opt.load_state_dict(st['optimizer']);updates=st['updates'];exposures=st['exposures'];torch.set_rng_state(st['CPU_rng']);torch.cuda.set_rng_state(st['CUDA_rng'])
  anchors=[]
  for state in [src['NN'],initial]:
   m=ATMS().cuda().eval().requires_grad_(False);m.load_state_dict(state,strict=True);anchors.append(m)
  visual=torch.load(K.checked(pq['visual_target']),map_location='cpu',weights_only=True,mmap=True)
  Qdone=K.read(Path(q['project_root'])/'Qbank/completion.json');assert Qdone['status']=='PASS_FOUR_T_PROBES_ALL70_LAYERS_FOUR_TIMES_EXACT_NATIVE_Q'
  Qbank=np.load(K.checked(Qdone['bank']));h94=c['cv']['bq']['h94'];K.checked(h94);w={};kv=[]
  with safe_open(h94['path'],framework='pt',device='cpu') as f:
   for key,name in [('P_weight','image_proj.proj.weight'),('P_bias','image_proj.proj.bias'),('P_gamma','image_proj.norm.weight'),('P_beta','image_proj.norm.bias')]:w[key]=f.get_tensor(name).half().float().cuda()
   for lay in Qdone['layouts']:
    idx=lay['ip_index'];kv.append({k:f.get_tensor(f'ip_adapter.{idx}.{n}.weight').half().float().cuda() for k,n in [('K','to_k_ip'),('V','to_v_ip')]})
  rng=np.random.default_rng(1729);schedule=np.concatenate([rng.permutation(280) for _ in range(3)])[:830];counts=np.bincount(schedule,minlength=280);assert counts.min()==2 and counts.max()==3
  for epoch in range(a.end-5,a.end):
   order=np.random.default_rng(1739+epoch).permutation(1323);coverage=np.zeros(13230,np.uint8);loss_sum=norm_sum=0.;ranks=[];energies=[];maxres=0.;t0=time.monotonic()
   for bi,lo in enumerate(range(0,1323,16)):
    guard();pos=(order[lo:lo+16,None]*10+np.arange(10)).reshape(-1);ids=tc['T'][pos];coverage[pos]+=1;n=len(ids);x=mean[ids].cuda();y=visual[ids].cuda();subject=torch.ones(n,device='cuda',dtype=torch.long)
    unit=int(schedule[updates]);ti,li=divmod(unit,70);bank=Qbank[f'l{li:02d}'][:,ti];h,dim=bank.shape[1],bank.shape[-1]
    Q=torch.tensor(bank.transpose(1,0,2,3).reshape(1,h,8,dim).copy(),device='cuda').float();weights={**w,**kv[li]}
    with torch.no_grad():
     ac=anchors[0](x,subject);bc=anchors[1](x,subject);old=H.response(ac,Q,weights);updated=H.response(bc,Q,weights);teachers=H.response(y,Q,weights)
     random=torch.randn(updated.shape,generator=torch.Generator(device='cuda').manual_seed(21729+updates),device='cuda')
     targets,scale,proof=H.targets(old,updated,teachers,random)
    target=targets['main' if a.arm=='response_repair' else 'matched_direction'];opt.zero_grad(set_to_none=True)
    pred=net(x,subject);value=H.response(pred,Q,weights);loss=H.loss(value,target,scale);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(params,1.))
    assert np.isfinite(norm) and torch.isfinite(loss);opt.step();assert all(torch.isfinite(p).all() for p in params)
    if updates==0:
     np.savez_compressed(dest/'first_batch_fixture.npz',T_rows=ids,Q=Q.cpu().numpy(),NN_condition=ac.cpu().numpy(),CN_condition=bc.cpu().numpy(),visual_condition=y.cpu().numpy(),old_response=old.cpu().numpy(),updated_response=updated.cpu().numpy(),target=target.cpu().numpy(),scale=scale.cpu().numpy(),unit=np.array(unit),loss=np.array(float(loss)),preclip_norm=np.array(norm))
    updates+=1;exposures+=n;loss_sum+=float(loss)*n;norm_sum+=norm;ranks.append(proof['rank']);energies.append(proof['mean_retained_energy_fraction']);maxres=max(maxres,proof['teacher_contrast_relative_residual'])
    if bi%16==0:K.write(dest/'progress.json',dict(epoch=epoch+1,batch=bi+1,updates=updates,exposures=exposures,online_loss=float(loss)))
   assert np.all(coverage==1)
   for n,v in net.named_buffers():assert torch.equal(v.cpu(),initial[n])
   for n,p in net.named_parameters():
    if not p.requires_grad:assert torch.equal(p.cpu(),initial[n]) and p.grad is None
   event=dict(epoch=epoch+1,updates=updates,exposures=exposures,coverage_min=int(coverage.min()),coverage_max=int(coverage.max()),concept_order_sha256=hashlib.sha256(order.tobytes()).hexdigest(),online_response_loss=loss_sum/13230,mean_preclip_norm=norm_sum/83,teacher_rank_min=min(ranks),teacher_rank_max=max(ranks),mean_retained_energy_fraction=float(np.mean(energies)),maximum_contrast_residual=maxres,head_logit_BN_frozen=True,seconds=time.monotonic()-t0)
   with (dest/'epochs.jsonl').open('a') as f:f.write(json.dumps(event)+'\n');f.flush();os.fsync(f.fileno())
   print(json.dumps(event),flush=True)
  assert updates==83*a.end and exposures==13230*a.end;save(dest/f'epoch{a.end:02d}.pt',state_record())
  result=dict(status='COMPLETE_RESPONSE_REPAIR_FULL_T_EPOCH_BOUNDARY',arm=a.arm,epoch=a.end,updates=updates,exposures=exposures,checkpoint=K.ref(dest/f'epoch{a.end:02d}.pt'),all280_units_min_exposures=int(counts.min()),schedule_sha256=hashlib.sha256(schedule.tobytes()).hexdigest());result_path=dest/f'fit_{a.end}_completion.json'
 else:
  test=torch.load(K.checked(tc['R80']['EEG_R80']),map_location='cpu',weights_only=True,mmap=True);ids=[r['array_row'] for r in tc['queries']];values=np.empty((600,2,1024),np.float32);models=[]
  for j,arm in enumerate(C.ARMS[2:]):
   done=K.read(out/arm/'fit_10_completion.json');assert done['updates']==830;st=torch.load(K.checked(done['checkpoint']),map_location='cpu',weights_only=True,mmap=True)
   net=ATMS().cuda();net.load_state_dict(st['model'],strict=True);net.eval()
   with torch.inference_mode():
    for lo in range(0,600,50):
     guard();x=mean[ids[lo:lo+50]] if lo<400 else test[lo-400:lo-350];values[lo:lo+50,j]=net(x.cuda(),torch.ones(len(x),device='cuda',dtype=torch.long)).cpu().numpy()
   models.append(done['checkpoint']);del net
  assert np.isfinite(values).all() and not (out/'conditions.npy').exists();np.save(out/'conditions.npy',values)
  result=dict(status='COMPLETE_RESPONSE_REPAIR_EEG_ONLY1200_CONDITIONS',conditions=K.ref(out/'conditions.npy'),models=models,normal_inference_only_EEG=True);result_path=out/'conditions_completion.json'
 guard();result.update(seconds=time.monotonic()-started,extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0,peak_GPU_bytes=torch.cuda.max_memory_allocated());K.write(result_path,result)
except BaseException as exc:
 partial=None
 if a.mode=='fit' and all(k in globals() for k in ['net','opt','epoch','updates','exposures']):
  try:save(dest/f'failed_end{a.end}.pt',state_record());partial=K.ref(dest/f'failed_end{a.end}.pt')
  except BaseException as err:partial=dict(error=repr(err))
 K.write(out/(suffix+'_failure.json'),dict(error=repr(exc),traceback=traceback.format_exc(),partial=partial));raise
