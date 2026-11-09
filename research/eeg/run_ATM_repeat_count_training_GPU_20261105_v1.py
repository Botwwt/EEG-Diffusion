"""Actual matched T10 training; no test-target input or architectural change."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import argparse,copy,fcntl,hashlib,json,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import atm_repeat_count_training_contract_20261105_v1 as C
import atm_repeat_count_training_core_20261105_v1 as F
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
  view_exposures=15*exposures,parent_checkpoint=q['checkpoint'],policy=K.ref(args.policy),CPU_rng=torch.get_rng_state(),CUDA_rng=torch.cuda.get_rng_state())
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
 if args.mode=='profile':
  math=F.math_checks();net,opt=loadnet(base);net.train();meta=G.F.metadata(net);ids=T[:160];block=raw[ids].astype(np.float32)
  assert np.array_equal(block.mean(1),mean[ids].numpy())
  torch.cuda.manual_seed(1729);pred=net(torch.tensor(block.mean(1),device='cuda'),torch.ones(160,device='cuda',dtype=torch.long))
  logits=net.logit_scale.exp().clamp_max(100)*torch.nn.functional.normalize(pred,dim=1)@torch.nn.functional.normalize(torch.tensor(target[ids],device='cuda'),dim=1).T
  ls,_=torch_loss(logits,'nce_continued');gs=torch.autograd.grad(ls,meta['parameters'],allow_unused=True)
  native=torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(meta['parameters'],gs)]).double()
  native_buffers={n:b.detach().clone() for n,b in net.named_buffers()};reference=numpy_loss_gradient(logits.detach().cpu().numpy(),'nce_continued')
  assert abs(float(ls)-reference['loss'])<3e-5
  net.load_state_dict(base['model'],strict=True);g,losses,active=F.collect_training_gradients(net,meta,block,target[ids],1729,torch_loss,guard)
  error=float((g[3]-native).abs().max());assert error==0
  assert all(torch.equal(b,native_buffers[n]) for n,b in net.named_buffers())
  R=G.optimizer_reference(meta,base,'cuda');proof=[];samples={};gram=(g@g.T).detach().cpu().numpy()
  for arm in F.ARMS:
   net.load_state_dict(base['model'],strict=True);opt.load_state_dict(copy.deepcopy(base['optimizer']))
   info=F.combine_and_assign(net,meta,g,losses,active,arm)
   rawg=sum(float(w)*v for w,v in zip(F.weights(arm),g));_,data,total=G.adam_vectors(rawg,meta,np.array(active),R)
   opt.step();actual=torch.cat([p.detach().reshape(-1).double() for p in meta['parameters']]);err=float((actual-R['theta']-total).abs().max());assert err<3e-6
   info.update(arm=arm,full_parameter_Adam_error=err,temporary_step_discarded=True);proof.append(info)
   samples[arm]=rawg[torch.as_tensor(meta['indices'],device='cuda')].cpu().numpy()
  net.load_state_dict(base['model'],strict=True)
  for n,v in net.state_dict().items():assert torch.equal(v.cpu(),base['model'][n])
  np.savez_compressed(out/'profile_fixture.npz',rows=ids,losses=losses,gradient_gram=gram,sampled_parameter_indices=meta['indices'],**samples)
  result=dict(status='PASS_THREE_RISKS_NATIVE_R4_EXACT_BN_ONCE_AND_REAL_ADAM',proof=proof,math=math,native_R4_gradient_max_error=error,
    all_buffers_equal_one_native_R4_forward=True,temporary_updates=3,new_final_models=0,all_original_state_restored=True)
 elif args.mode=='fit':
  assert args.arm in F.ARMS and args.end in (5,10)
  assert K.read(out/'profile_completion.json')['status']=='PASS_THREE_RISKS_NATIVE_R4_EXACT_BN_ONCE_AND_REAL_ADAM'
  directory=out/'fits'/args.arm;directory.mkdir(parents=True,exist_ok=True);destination=directory/f'epoch{args.end:02d}.pt';assert not destination.exists()
  initial=base if args.end==5 else torch.load(directory/'epoch05.pt',map_location='cpu',weights_only=True,mmap=True)
  net,opt=loadnet(initial);net.train();meta=G.F.metadata(net);updates=0 if args.end==5 else initial['updates'];exposures=0 if args.end==5 else initial['exposures']
  for epoch in range(args.end-5,args.end):
   order=np.random.default_rng(1739+epoch).permutation(1323);coverage=np.zeros(13230,np.uint8);sums=np.zeros(4);risks=norms=cosines=steps=0.;seen=0
   for bi,lo in enumerate(range(0,1323,16)):
    guard();groups=order[lo:lo+16];pos=(groups[:,None]*10+np.arange(10)).reshape(-1);ids=T[pos];block=raw[ids].astype(np.float32)
    assert np.array_equal(block.mean(1),mean[ids].numpy());coverage[pos]+=1
    g,losses,active=F.collect_training_gradients(net,meta,block,target[ids],101729+epoch*10000+bi,torch_loss,guard)
    oldpars=torch.cat([p.detach().reshape(-1) for p in meta['parameters']]);info=F.combine_and_assign(net,meta,g,losses,active,args.arm);opt.step()
    assert all(torch.isfinite(p).all() for p in meta['parameters'])
    step=float(torch.linalg.vector_norm(torch.cat([p.detach().reshape(-1) for p in meta['parameters']])-oldpars));sums+=losses*len(ids)
    risks+=info['risk']*len(ids);norms+=info['preclip_norm'];cosines+=info['gradient_cosine_vs_own_R4'];steps+=step;seen+=len(ids);updates+=1;exposures+=len(ids)
    if bi%16==0:K.write(out/'progress.json',dict(phase='FULL_T_MATCHED_COUNT_RISK',arm=args.arm,epoch=epoch+1,batch=bi+1,updates=updates,exposures=exposures))
   assert np.all(coverage==1) and seen==13230
   event=dict(arm=args.arm,epoch=epoch+1,updates=updates,exposures=exposures,view_exposures=15*exposures,concept_order_sha256=hashlib.sha256(order.tobytes()).hexdigest(),coverage_min=int(coverage.min()),coverage_max=int(coverage.max()),
    online_mean_losses_by_count=(sums/seen).tolist(),online_risk=risks/seen,mean_preclip_norm=norms/83,mean_gradient_cosine_vs_own_R4=cosines/83,mean_parameter_step_norm=steps/83,
    effective_logit_scale=float(net.logit_scale.exp().clamp_max(100).detach()),R80_data_fitted=False,Dfit_or_Dcheck_used=False,seconds=time.monotonic()-start)
   with (directory/'epochs.jsonl').open('a') as f:f.write(json.dumps(event)+'\n');f.flush();os.fsync(f.fileno())
   print(json.dumps(event),flush=True)
  assert updates==args.end*83 and exposures==args.end*13230
  save(destination,checkpoint(net,opt,args.end,args.arm,updates,exposures))
  result=dict(status='COMPLETE_MATCHED_COUNT_RISK_T_EPOCH_BOUNDARY',arm=args.arm,epoch=args.end,updates=updates,exposures=exposures,view_exposures=15*exposures,checkpoint=K.ref(destination))
  K.write(directory/f'completion{args.end:02d}.json',result)
 else:
  mean=torch.load(pq['EEG']['path'],map_location='cpu',weights_only=True,mmap=True)
  test=torch.load(K.checked(c['R80']['EEG_R80']),map_location='cpu',weights_only=True,mmap=True);assert test.shape==(200,63,250)
  ids=[r['array_row'] for r in c['queries']];values=np.empty((600,4,1024),np.float32)
  for j,arm in enumerate(('frozen_positive_NCE',)+F.ARMS):
   state=base if j==0 else torch.load(K.checked(K.read(out/'fits'/arm/'completion10.json')['checkpoint']),map_location='cpu',weights_only=True,mmap=True)
   net,opt=loadnet(state);del opt;net.eval()
   with torch.inference_mode():
    for lo in range(0,600,100):
     guard();x=mean[ids[lo:lo+100]] if lo<400 else test[lo-400:lo-300]
     values[lo:lo+100,j]=net(x.cuda(),torch.ones(len(x),device='cuda',dtype=torch.long)).cpu().numpy()
   del net;torch.cuda.empty_cache()
  assert np.isfinite(values).all();np.save(out/'conditions_R4_400_R80_200.npy',values)
  K.write(out/'R4_queries.json',c['queries']);K.write(out/'R80_queries.json',K.read(K.checked(c['R80']['identity_table'])))
  result=dict(status='COMPLETE_EEG_ONLY_ALL2400_CONDITIONS',conditions=K.ref(out/'conditions_R4_400_R80_200.npy'),R4_queries=K.ref(out/'R4_queries.json'),R80_queries=K.ref(out/'R80_queries.json'),arms=['frozen_positive_NCE']+list(F.ARMS),normal_inference_only_EEG=True)
 guard();result.update(seconds=time.monotonic()-start,peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0)
 K.write(out/(suffix+'_completion.json'),result)
except BaseException as exc:
 partial=None
 if args.mode=='fit' and 'net' in globals() and 'opt' in globals():
  try:
   pp=out/'fits'/args.arm/f'failed_end{args.end:02d}.pt';save(pp,checkpoint(net,opt,epoch,args.arm,updates,exposures));partial=K.ref(pp)
  except BaseException as pe:partial=dict(error=repr(pe))
 K.write(out/(suffix+'_failure.json'),dict(error=repr(exc),traceback=traceback.format_exc(),partial=partial));raise
