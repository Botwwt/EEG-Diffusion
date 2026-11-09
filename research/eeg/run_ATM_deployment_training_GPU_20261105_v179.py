"""Full allowed T, saved Adam continuation; two fixed targets and immutable parents."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import argparse,copy,fcntl,hashlib,json,resource,subprocess,sys,time,traceback
from pathlib import Path
import atm_deployment_training_contract_20261105_v179 as C
import atm_deployment_supervision_core_20261105_v3 as F
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--mode',choices=['fit','conditions'],required=True);ap.add_argument('--arm',default='');ap.add_argument('--end',type=int,default=0);args=ap.parse_args()
started=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
c=C.setup(args.policy);q,K,pq,out=c['q'],c['K'],c['physical'],c['out'];before=C.resources(c,starting=True);assert K.ref(__file__)==q['sources']['GPU']
repo=Path(pq['GPU_wrapper']['path']).parent;stage=args.mode+('_'+args.arm+'_'+str(args.end) if args.mode=='fit' else '')
j=K.read(repo/'audit'/(q['job_prefix']+'_'+stage+'_job.json'));assert j['status']=='RUNNING' and j['pid']==os.getpid()
lock=(repo/'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==pq['GPU_UUID']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
sys.path[:0]=pq['existing_runtime_paths']+['/root/atm_official_20261016']
import numpy as np
import torch
import torch.nn.functional as TF
from models.atms import ATMS
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(1729)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(pq['maximum_GPU_bytes']/torch.cuda.get_device_properties(0).total_memory)
def guard():
 assert time.monotonic()-started<850
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<pq['maximum_extra_CPU_bytes']
 assert torch.cuda.max_memory_allocated()<pq['maximum_GPU_bytes']
 used=K.allocated(q['project_root']);assert used<q['maximum_output_bytes']
 st=os.statvfs(q['project_root']);assert st.f_bavail*st.f_frsize>=before['required_including_this_project']+before['own_used_bytes']-used
def save(path,state):
 assert not path.exists();tmp=path.with_name(path.name+'.pending');assert not tmp.exists();torch.save(state,tmp);tmp.replace(path)
def state(net,opt,epoch,updates,exposures):
 op=copy.deepcopy(opt.state_dict())
 for z in op['state'].values():
  for k,v in z.items():
   if torch.is_tensor(v):z[k]=v.cpu()
 return dict(model={n:v.detach().cpu().clone() for n,v in net.state_dict().items()},optimizer=op,
 epoch=epoch,updates=updates,Adam_updates=830+updates,exposures=exposures,EEG_forward_exposures=2*exposures,arm=args.arm,
 parent=q['checkpoint'],policy=K.ref(args.policy),CPU_rng=torch.get_rng_state(),CUDA_rng=torch.cuda.get_rng_state())
def loadnet(ck):
 net=ATMS().cuda();net.load_state_dict(ck['model'],strict=True);opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=.01)
 opt.load_state_dict(copy.deepcopy(ck['optimizer']));return net,opt
try:
 base=torch.load(K.checked(q['checkpoint']),map_location='cpu',weights_only=True,mmap=True)
 assert base['updates']==830 and base['epoch']==10 and base['arm']=='nce_continued'
 mean=torch.load(K.checked(pq['EEG']),map_location='cpu',weights_only=True,mmap=True)
 if args.mode=='fit':
  assert args.arm in C.ARMS and args.end in (5,10)
  vis=torch.load(K.checked(pq['visual_target']),map_location='cpu',weights_only=True,mmap=True)
  z=np.load(K.checked(q['projector']))
  W=torch.from_numpy(z['image_embeds.weight'].astype(np.float32)).cuda();b=torch.from_numpy(z['image_embeds.bias'].astype(np.float32)).cuda()
  g=torch.from_numpy(z['norm.weight'].astype(np.float32)).cuda();be=torch.from_numpy(z['norm.bias'].astype(np.float32)).cuda()
  def project(x):return TF.layer_norm(TF.linear(x,W,b).reshape(-1,4,2048),(2048,),g,be,1e-5)
  directory=out/'fits'/args.arm;directory.mkdir(parents=True,exist_ok=True);dest=directory/f'epoch{args.end:02d}.pt';assert not dest.exists()
  initial=base if args.end==5 else torch.load(directory/'epoch05.pt',map_location='cpu',weights_only=True,mmap=True)
  net,opt=loadnet(initial);updates=0 if args.end==5 else initial['updates'];exposures=0 if args.end==5 else initial['exposures']
  bn_start={n:int(v) for n,v in net.named_buffers() if n.endswith('num_batches_tracked')}
  T=np.array(q['T_array_rows'],np.int64)
  assert hashlib.sha256(T.tobytes()).hexdigest()==q['T_rows_sha256']
  for epoch in range(args.end-5,args.end):
   order=np.random.default_rng(1739+epoch).permutation(1323);coverage=np.zeros(13230,np.uint8);sums={};seen=0
   for bi,lo in enumerate(range(0,1323,16)):
    guard();pos=(order[lo:lo+16,None]*10+np.arange(10)).ravel();ids=T[pos];coverage[pos]+=1
    x=mean[ids].cuda();y=vis[ids].cuda();grads,info=F.collect(net,x,y,project,args.arm,101729+epoch*10000+bi)
    norm=F.apply_step(net,opt,grads);updates+=1;exposures+=len(ids);seen+=len(ids)
    assert all(int(s['step'])==830+updates for s in opt.state.values())
    for key in ['CE','P','mapper_CE_norm','mapper_P_norm','mapper_dot','mapper_cosine']:
     sums[key]=sums.get(key,0.)+info[key]*len(ids)
    sums['gradient_norm_before_clip']=sums.get('gradient_norm_before_clip',0.)+norm*len(ids)
    assert all(int(v)-bn_start[n]==updates-(args.end-5)*83 for n,v in net.named_buffers() if n in bn_start)
    if bi%16==0:K.write(out/'progress.json',dict(phase='FULL_T_DEPLOYMENT_SUPERVISION',arm=args.arm,epoch=epoch+1,batch=bi+1,updates=updates,exposures=exposures))
    del grads,x,y
   assert seen==13230 and np.all(coverage==1)
   event=dict(arm=args.arm,epoch=epoch+1,updates=updates,exposures=exposures,EEG_forward_exposures=2*exposures,
    coverage_min=1,coverage_max=1,concept_order_sha256=hashlib.sha256(order.tobytes()).hexdigest(),online_mean={k:v/seen for k,v in sums.items()},
    BN_updates=83,EEG_forward_calls=166,derivative_calls=166,effective_logit_scale=float(net.logit_scale.exp().clamp_max(100).detach()),
    lambda_P=1.,D_or_R80_fit=False,seconds=time.monotonic()-started)
   with (directory/'epochs.jsonl').open('a') as f:f.write(json.dumps(event)+'\n');f.flush();os.fsync(f.fileno())
   print(json.dumps(event),flush=True)
  assert updates==args.end*83 and exposures==args.end*13230
  save(dest,state(net,opt,args.end,updates,exposures))
  result=dict(status='COMPLETE_DEPLOYMENT_SUPERVISION_T_EPOCH_BOUNDARY',arm=args.arm,epoch=args.end,updates=updates,exposures=exposures,checkpoint=K.ref(dest))
  K.write(directory/f'completion{args.end:02d}.json',result)
 else:
  old=c['old'];test=torch.load(K.checked(old['train']['R80']['EEG_R80']),map_location='cpu',weights_only=True,mmap=True)
  queries=old['queries'];ids=[r['array_row'] for r in queries['R4']];assert len(ids)==400 and test.shape==(200,63,250)
  values=np.empty((600,4,1024),np.float32)
  for j,arm in enumerate(C.ALL_ARMS):
   pin=q['reference_checkpoints'][arm] if arm in q['reference_checkpoints'] else K.read(out/'fits'/arm/'completion10.json')['checkpoint']
   ck=torch.load(K.checked(pin),map_location='cpu',weights_only=True,mmap=True);net,opt=loadnet(ck);del opt;net.eval()
   with torch.inference_mode():
    for lo in range(0,600,100):
     guard();x=mean[ids[lo:lo+100]] if lo<400 else test[lo-400:lo-300]
     values[lo:lo+100,j]=net(x.cuda(),torch.ones(len(x),device='cuda',dtype=torch.long)).cpu().numpy()
   del net;torch.cuda.empty_cache()
  assert np.isfinite(values).all();np.save(out/'conditions_R4_400_R80_200.npy',values)
  K.write(out/'queries.json',queries)
  result=dict(status='COMPLETE_EEG_ONLY_ALL2400_CONDITIONS',conditions=K.ref(out/'conditions_R4_400_R80_200.npy'),queries=K.ref(out/'queries.json'),arms=list(C.ALL_ARMS),normal_inference_only_EEG=True)
 guard();K.checked(q['checkpoint'])
 result.update(seconds=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0)
 K.write(out/(stage+'_completion.json'),result)
except BaseException as exc:
 partial=None
 if args.mode=='fit' and 'net' in globals() and 'opt' in globals():
  try:
   p=out/'fits'/args.arm/f'failed_end{args.end:02d}.pt';save(p,state(net,opt,epoch,updates,exposures));partial=K.ref(p)
  except BaseException as error:partial=dict(error=repr(error))
 K.write(out/(stage+'_failure.json'),dict(error=repr(exc),traceback=traceback.format_exc(),partial=partial));raise
