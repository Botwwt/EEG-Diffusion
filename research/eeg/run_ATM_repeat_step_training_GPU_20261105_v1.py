"""Actual matched T10 training; no test-target input or architectural change."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import argparse,copy,fcntl,hashlib,json,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import atm_repeat_step_training_contract_20261105_v1 as C
import atm_repeat_step_training_core_20261105_v1 as F
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--mode',required=True,choices=['profile','fit','conditions']);ap.add_argument('--arm',default='');ap.add_argument('--end',type=int,default=0);args=ap.parse_args()
start=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
c=C.setup(args.policy);q,K,out,pq=c['q'],c['K'],c['out'],c['physical'];assert K.ref(__file__)==q['sources']['GPU'];before=C.resources(c,starting=True)
repo=Path(pq['GPU_wrapper']['path']).parent;suffix=args.mode+('_'+args.arm+'_'+str(args.end) if args.mode=='fit' else '')
job=K.read(repo/'audit'/(q['job_prefix']+'_'+suffix+'_job.json'));assert job['pid']==os.getpid() and job['status']=='RUNNING'
lock=(repo/'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==pq['GPU_UUID']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
sys.path[:0]=pq['existing_runtime_paths']+['/root/atm_official_20261016']
import torch
from models.atms import ATMS
from atm_conditional_stimulus_objective_20261104_v1 import torch_loss,numpy_loss_gradient
import atm_actual_NCE_Adam_repeat_core_20261105_v1 as G
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(1729)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(pq['maximum_GPU_bytes']/torch.cuda.get_device_properties(0).total_memory)
def guard():
 assert time.monotonic()-start<850
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<pq['maximum_extra_CPU_bytes']
 assert torch.cuda.max_memory_allocated()<pq['maximum_GPU_bytes']
 used=K.allocated(q['project_root']);assert used<q['maximum_output_bytes']
 st=os.statvfs(q['project_root']);assert st.f_bavail*st.f_frsize>=before['required_including_this_project']+before['own_used_bytes']-used
def save(path,state):
 assert not path.exists();tmp=path.with_name(path.name+'.pending');assert not tmp.exists();torch.save(state,tmp);tmp.replace(path)
def checkpoint(net,opt,epoch,arm,updates,exposures):
 op=copy.deepcopy(opt.state_dict())
 for state in op['state'].values():
  for key,value in state.items():
   if torch.is_tensor(value):state[key]=value.cpu()
 return dict(model={n:v.detach().cpu().clone() for n,v in net.state_dict().items()},optimizer=op,epoch=epoch,arm=arm,updates=updates,exposures=exposures,
  view_exposures=5*exposures,parent_checkpoint=q['checkpoint'],policy=K.ref(args.policy),CPU_rng=torch.get_rng_state(),CUDA_rng=torch.cuda.get_rng_state())
def loadnet(state):
 net=ATMS().cuda();net.load_state_dict(state['model'],strict=True);opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=.01)
 opt.load_state_dict(copy.deepcopy(state['optimizer']));return net,opt
try:
 base=torch.load(K.checked(q['checkpoint']),map_location='cpu',weights_only=True,mmap=True);assert base['updates']==830 and base['epoch']==10
 if args.mode in ('profile','fit'):
  obj=np.load(K.checked(pq['physical_source']),allow_pickle=True)
  if not isinstance(obj,dict):obj=obj.item()
  raw=obj['preprocessed_eeg_data'];assert raw.shape==(16540,4,63,250) and raw.dtype==np.float32
  mean=torch.load(pq['EEG']['path'],map_location='cpu',weights_only=True,mmap=True)
  target=torch.load(pq['visual_target']['path'],map_location='cpu',weights_only=True,mmap=True).numpy();T=c['T']
 if args.mode=='fit':
  assert args.arm in F.ARMS and args.end in (5,10)
  assert K.read(K.checked(q['profile_completion']))['status']=='COMPLETE_REAL_TRAIN_REPEAT_STEP_PROFILE'
  directory=out/'fits'/args.arm;directory.mkdir(parents=True,exist_ok=True);destination=directory/f'epoch{args.end:02d}.pt';assert not destination.exists()
  initial=base if args.end==5 else torch.load(directory/'epoch05.pt',map_location='cpu',weights_only=True,mmap=True)
  net,opt=loadnet(initial);net.train();meta=G.F.metadata(net);updates=0 if args.end==5 else initial['updates'];exposures=0 if args.end==5 else initial['exposures']
  signs=F.signs_for(meta,'cuda')
  for epoch in range(args.end-5,args.end):
   order=np.random.default_rng(1739+epoch).permutation(1323);coverage=np.zeros(13230,np.uint8);sums=np.zeros(5);totals={};seen=0
   for bi,lo in enumerate(range(0,1323,16)):
    guard();groups=order[lo:lo+16];pos=(groups[:,None]*10+np.arange(10)).reshape(-1);ids=T[pos];block=raw[ids].astype(np.float32)
    assert np.array_equal(block.mean(1),mean[ids].numpy());coverage[pos]+=1
    g,losses,active=F.collect(net,meta,block,target[ids],101729+epoch*10000+bi,torch_loss,guard)
    provisional=F.provisional_Adam(net,opt,meta,g,active);info,_=F.finish(net,meta,g,active,provisional,args.arm,signs)
    assert info['Adam_step']==831+updates
    for key,value in info.items():
     if key!='arm':totals[key]=totals.get(key,0.)+value
    sums+=losses*len(ids);seen+=len(ids);updates+=1;exposures+=len(ids)
    if bi%16==0:K.write(out/'progress.json',dict(phase='FULL_T_PAIRED_REPEAT_STEP',arm=args.arm,epoch=epoch+1,batch=bi+1,updates=updates,exposures=exposures))
   assert np.all(coverage==1) and seen==13230
   event=dict(arm=args.arm,epoch=epoch+1,updates=updates,exposures=exposures,view_exposures=5*exposures,concept_order_sha256=hashlib.sha256(order.tobytes()).hexdigest(),coverage_min=int(coverage.min()),coverage_max=int(coverage.max()),
    online_mean_losses_R1_0_1_2_3_R4=(sums/seen).tolist(),mean_step_diagnostics={k:v/83 for k,v in totals.items()},
    effective_logit_scale=float(net.logit_scale.exp().clamp_max(100).detach()),R80_data_fitted=False,Dfit_or_Dcheck_used=False,seconds=time.monotonic()-start)
   with (directory/'epochs.jsonl').open('a') as f:f.write(json.dumps(event)+'\n');f.flush();os.fsync(f.fileno())
   print(json.dumps(event),flush=True)
  assert updates==args.end*83 and exposures==args.end*13230
  save(destination,checkpoint(net,opt,args.end,args.arm,updates,exposures))
  result=dict(status='COMPLETE_PAIRED_REPEAT_STEP_T_EPOCH_BOUNDARY',arm=args.arm,epoch=args.end,updates=updates,exposures=exposures,view_exposures=5*exposures,checkpoint=K.ref(destination))
  K.write(directory/f'completion{args.end:02d}.json',result)
 else:
  mean=torch.load(pq['EEG']['path'],map_location='cpu',weights_only=True,mmap=True)
  test=torch.load(K.checked(c['R80']['EEG_R80']),map_location='cpu',weights_only=True,mmap=True);assert test.shape==(200,63,250)
  ids=[r['array_row'] for r in c['queries']];values=np.empty((600,6,1024),np.float32)
  for j,arm in enumerate(tuple(q['reference_checkpoints'])+F.ARMS):
   pin=q['reference_checkpoints'][arm] if arm in q['reference_checkpoints'] else K.read(out/'fits'/arm/'completion10.json')['checkpoint']
   state=torch.load(K.checked(pin),map_location='cpu',weights_only=True,mmap=True)
   net,opt=loadnet(state);del opt;net.eval()
   with torch.inference_mode():
    for lo in range(0,600,100):
     guard();x=mean[ids[lo:lo+100]] if lo<400 else test[lo-400:lo-300]
     values[lo:lo+100,j]=net(x.cuda(),torch.ones(len(x),device='cuda',dtype=torch.long)).cpu().numpy()
   del net;torch.cuda.empty_cache()
  assert np.isfinite(values).all();np.save(out/'conditions_R4_400_R80_200.npy',values)
  K.write(out/'R4_queries.json',c['queries']);K.write(out/'R80_queries.json',K.read(K.checked(c['R80']['identity_table'])))
  result=dict(status='COMPLETE_EEG_ONLY_ALL3600_CONDITIONS',conditions=K.ref(out/'conditions_R4_400_R80_200.npy'),R4_queries=K.ref(out/'R4_queries.json'),R80_queries=K.ref(out/'R80_queries.json'),arms=list(q['reference_checkpoints'])+list(F.ARMS),normal_inference_only_EEG=True)
 guard();result.update(seconds=time.monotonic()-start,peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0)
 K.write(out/(suffix+'_completion.json'),result)
except BaseException as exc:
 partial=None
 if args.mode=='fit' and 'net' in globals() and 'opt' in globals():
  try:
   pp=out/'fits'/args.arm/f'failed_end{args.end:02d}.pt';save(pp,checkpoint(net,opt,epoch,args.arm,updates,exposures));partial=K.ref(pp)
  except BaseException as pe:partial=dict(error=repr(pe))
 K.write(out/(suffix+'_failure.json'),dict(error=repr(exc),traceback=traceback.format_exc(),partial=partial));raise
