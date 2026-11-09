"""Native positive ATMS, original saved Adam, two real T batches, discarded steps."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import argparse,copy,datetime,fcntl,hashlib,json,resource,subprocess,sys,time
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);args=ap.parse_args()
q=json.loads(Path(args.policy).read_bytes());assert q['status']=='LOCKED_DEPLOYMENT_SUPERVISION_REAL_PROFILE_ONLY'
A=Path(q['root']);O=A/'output';O.mkdir(exist_ok=True);start=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def check(r):assert sha(r['path'])==r['sha256'];return r['path']
assert sha(__file__)==q['sources']['GPU']['sha256']
repo=Path(q['physical']['GPU_wrapper']['path']).parent
job=json.loads((repo/'audit'/(q['job_prefix']+'_job.json')).read_bytes());assert job['pid']==os.getpid() and job['status']=='RUNNING'
lock=(repo/'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==q['physical']['GPU_UUID']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
sys.path[:0]=[str(A)]+q['existing_runtime_paths']+['/root/atm_official_20261016']
import numpy as np
import torch
import torch.nn.functional as F
from models.atms import ATMS
import atm_deployment_supervision_core_20261105_v1 as C
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(q['physical']['maximum_GPU_bytes']/torch.cuda.get_device_properties(0).total_memory)
def guard():
 assert time.monotonic()-start<850
 assert torch.cuda.max_memory_allocated()<q['physical']['maximum_GPU_bytes']
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<q['physical']['maximum_extra_CPU_bytes']
 assert sum(p.stat().st_size for p in A.rglob('*') if p.is_file())<q['maximum_bytes']
def event(v):print(json.dumps(v),flush=True)
def checksum(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
for r in q['model_source_pins']:check(r)
for r in q['sources'].values():check(r)
raw=torch.load(check(q['EEG']),map_location='cpu',weights_only=True,mmap=True)
vis=torch.load(check(q['visual_target']),map_location='cpu',weights_only=True,mmap=True)
base=torch.load(check(q['encoder']),map_location='cpu',weights_only=True,mmap=True)
assert base['updates']==830 and base['epoch']==10 and base['arm']=='nce_continued'
z=np.load(check(q['projector']));W=torch.from_numpy(z['image_embeds.weight'].astype(np.float32)).cuda();b=torch.from_numpy(z['image_embeds.bias'].astype(np.float32)).cuda();gm=torch.from_numpy(z['norm.weight'].astype(np.float32)).cuda();bt=torch.from_numpy(z['norm.bias'].astype(np.float32)).cuda()
def project(x):return F.layer_norm(F.linear(x,W,b).reshape(-1,4,2048),(2048,),gm,bt,1e-5)
def model():
 n=ATMS().cuda();n.load_state_dict(base['model'],strict=True);o=torch.optim.AdamW(n.parameters(),lr=1e-4,weight_decay=.01);o.load_state_dict(copy.deepcopy(base['optimizer']))
 assert all(int(s['step'])==830 for s in o.state.values())
 return n,o
def flat(n):return torch.cat([p.detach().reshape(-1) for p in n.parameters()])
def eval_values(n,x,y):
 n.eval()
 with torch.no_grad():
  c=n(x,torch.ones(len(x),device='cuda',dtype=torch.long));p=(project(c)-project(y)).square().mean();ce=C.symmetric_ce(n,c,y)
 return c.detach(),torch.stack([p,ce]).cpu().numpy()
T=np.asarray(q['T_array_rows'],np.int64);order=(np.random.default_rng(1739).permutation(1323)[:,None]*10+np.arange(10)).ravel();batches=[order[i:i+160] for i in range(0,13230,160)]
results=[];arrays={};proofs=[];updates=0
for bi in q['batches']:
 ids=T[batches[bi]];seed=101729+bi;x=raw[ids].cuda();y=vis[ids].cuda();arrays[f'b{bi}__ids']=ids;arrays[f'b{bi}__visual']=y.cpu().numpy()
 for arm in C.ARMS:
  guard();n,o=model();pars=list(n.parameters());initial=flat(n).clone()
  pred0,initial_metrics=eval_values(n,x,y)
  captured=[]
  def recording_project(c):
   token=project(c)
   if c.requires_grad:captured.append(c.detach().cpu().numpy())
   return token
  grads,info=C.collect(n,x,y,recording_project,arm,seed);assert len(captured)==1
  arrays[f'b{bi}__{arm}__P_input']=captured[0]
  bn_reference={k:v.detach().clone() for k,v in n.named_buffers()}
  before_step,before_metrics=eval_values(n,x,y)
  norm=C.apply_step(n,o,grads);updates+=1
  after=flat(n);step=after-initial
  pred1,after_metrics=eval_values(n,x,y)
  step831=all(int(s['step'])==831 for s in o.state.values());assert step831
  assert all(torch.equal(v,bn_reference[k]) for k,v in n.named_buffers())
  chosen_idx=torch.tensor(q['gradient_indices'],device='cuda')
  arrays[f'b{bi}__{arm}__step_coordinates']=step[chosen_idx].cpu().numpy()
  arrays[f'b{bi}__{arm}__gradient_coordinates']=torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(pars,grads)])[chosen_idx].cpu().numpy()
  arrays[f'b{bi}__{arm}__eval_before']=before_step.cpu().numpy();arrays[f'b{bi}__{arm}__eval_after']=pred1.cpu().numpy()
  canonical=None
  if arm in ('nce_control','stochastic_P'):
   canonical,oo=model();canonical.train();C.set_seed(x.device,seed);cc=canonical(x,torch.ones(len(x),device='cuda',dtype=torch.long))
   loss=C.symmetric_ce(canonical,cc,y)
   if arm=='stochastic_P':loss=loss+(project(cc)-project(y)).square().mean()
   oo.zero_grad(set_to_none=True);loss.backward()
   expected=[p.grad for p in canonical.parameters()]
   ge=max(float((a-g).abs().max()) for a,g in zip(expected,grads) if a is not None and g is not None)
   assert all((a is None)==(g is None) for a,g in zip(expected,grads))
   scale=max(float(a.abs().max()) for a in expected if a is not None);assert ge<=3e-6*max(scale,1.)
   torch.nn.utils.clip_grad_norm_(canonical.parameters(),1.);oo.step();updates+=1
   state_error=max(float((v-canonical.state_dict()[k]).abs().max()) for k,v in n.state_dict().items())
   if arm=='nce_control':
    assert ge==0 and all(torch.equal(v,canonical.state_dict()[k]) for k,v in n.state_dict().items())
    for p,cp in zip(n.parameters(),canonical.parameters()):
     assert set(o.state.get(p,{}))==set(oo.state.get(cp,{}))
     for k,v in o.state.get(p,{}).items():assert torch.equal(v,oo.state[cp][k])
   else:assert state_error<5e-6
   proofs.append(dict(batch=bi,arm=arm,canonical_gradient_max_error=ge,canonical_full_state_max_error=state_error,native_CE_step_bitexact=arm=='nce_control',canonical_Adam_step=831))
   del canonical,oo,cc,loss,expected
  row=dict(**info,batch=bi,count=len(ids),seed=seed,gradient_norm_before_clip=norm,Adam_step=831,saved_optimizer_moments_resumed=True,
   inference_initial=initial_metrics.tolist(),BN_matched_before=before_metrics.tolist(),inference_after=after_metrics.tolist(),BN_matched_change=(after_metrics-before_metrics).tolist(),
   actual_parameter_step_norm=float(step.double().norm()),step_SHA256=checksum(step),no_new_architecture_parameters=True)
  results.append(row);event(row)
  del n,o,grads,pars,pred0,pred1,before_step,after,step,initial
  torch.cuda.empty_cache()
 # Original full model file remains immutable; temporary candidates are not saved.
 guard()
assert updates==10 and len(results)==6 and len(proofs)==4;check(q['encoder']);guard()
np.savez_compressed(O/'fixtures.npz',**arrays)
result=dict(status='PASS_REAL_TWO_BATCH_THREE_ROUTE_PROFILE',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),policy=q,results=results,canonical_proofs=proofs,
 temporary_parameter_updates=10,new_fits=0,new_images=0,source_checkpoint_unchanged=True,discarded_temporary_models=True,lambda_P_fixed=1.,
 seconds=time.monotonic()-start,peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0,
 CPU_float64_P_audit_pending=True,no_automatic_fit_admission=True,NEW_contacted=False)
(O/'completion.json').write_text(json.dumps(result,indent=2)+'\n');event(dict(status=result['status'],seconds=result['seconds'],temporary_updates=10))
