"""Fixed full-T native-class conditional mixture ML and matched visual teacher.

The native generator is never optimized here. All image likelihoods are a
previously audited frozen T-only bank. The normal model input remains EEG.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import argparse,datetime,hashlib,json,shutil,subprocess,sys,time,traceback
from pathlib import Path
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))
def check(x):p=Path(x['path']);assert ref(p)==x;return p
def write(p,x,replace=False):
 pending=p.with_name(p.name+'.pending')
 with pending.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 if not replace:assert not p.exists()
 os.replace(pending,p)
def peak():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
def digest(state):
 import torch
 h=hashlib.sha256()
 for k,v in state.items():
  t=v.detach().cpu().contiguous();h.update(k.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
def run(sp,job,mode):
 s=json.loads(Path(sp).read_bytes());assert s['status']=='REGISTERED_NATIVE_CATEGORICAL_FULL_T_MAIN_CONTROL_v1' and s['arm'] in ('EEG_coupled_native_class','visual_only_native_class')
 assert s['recipe']==dict(seed=1729,epochs=10,batch=64,updates=2070,query_exposures=132300,lr=.0001,weight_decay=.01,tokens=5)
 for x in s['source_pins']:check(x)
 root=Path(s['output_root']);root.mkdir(exist_ok=True);out=root/(mode+'_'+job);out.mkdir();started=time.monotonic();base=peak();phase='EXCLUSIVE_FIXED_INPUTS'
 try:
  expected='run_gpu_job_v2.py' if s['host']=='old' else 'run_BrainHIVE_two_host_GPU_job_20261028_v1.py'
  assert expected in (Path('/proc')/str(os.getppid())/'cmdline').read_bytes().decode().replace('\0',' ')
  assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==s['GPU_UUID']
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  r=json.loads((Path(s['repository_root'])/'audit'/(job+'_job.json')).read_bytes());assert r['status']=='RUNNING' and r['pid']==os.getpid() and float(r['max_seconds'] if s['host']=='old' else r['maximum_seconds'])==900
  old=json.loads(check(s['base_EEG_spec']).read_bytes());sys.path.insert(0,old['training_code']);import AR_anchored_training_contract_CPU_20261029_v1 as C
  a,_=C.load_admission(old['training_admission']);C.validate_sources(a);C.verify_protected_files(a);sys.path.insert(0,a['vendor'])
  import numpy as np
  import torch
  from eeg_labram_pretrained_20261028_v1 import build_encoder
  from eeg_labram_interface_20261028_v1 import CacheNativeEEGTransform
  from EEG_native_categorical_core_20261029_v1 import EEGNativeClassPrior,conditional_loss
  torch.set_num_threads(2);torch.manual_seed(1729);torch.cuda.manual_seed_all(1729);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  def guard():
   assert time.monotonic()-started<800 and peak()-base<10<<30 and torch.cuda.max_memory_allocated()<20<<30
   assert shutil.disk_usage(root).free>=2<<30
  bank=json.loads(check(s['bank_proof']).read_bytes());assert bank['status']=='PASS_COMPLETE_T_NATIVE_CLASS_LIKELIHOOD_EXACT_SHARED_AND_INDEPENDENT64_REPLICATION' and bank['T_rows']==13230 and bank['native_classes']==1000
  assert bank['token_source_sha256']==a['data']['tokens']['sha256'] and bank['identities_sha256']==a['data']['token_identities']['sha256']
  ll=np.load(check(bank['likelihoods_by_host'][s['host']]),mmap_mode='r',allow_pickle=False);assert ll.shape==(13230,1000) and ll.dtype==np.float32 and np.isfinite(ll).all() and (ll<=0).all()
  raw=np.load(check(old['T_mean63']),mmap_mode='r',allow_pickle=False);assert raw.shape==(13230,63,250)
  ids=json.loads(check(a['data']['EEG_identities']).read_bytes());tids=json.loads(check(a['data']['token_identities']).read_bytes());C.identity_join(ids['T'],tids)
  stats=json.loads(check(old['full_EEG']['normalization']).read_bytes());assert stats['role']=='full' and stats['training_positions']==list(range(13230))
  tr=CacheNativeEEGTransform(tuple(stats['channel_names']),np.array(stats['center']),np.array(stats['scale']));N=64 if mode=='profile' else 13230;x=np.empty((N,63,1,200),np.float32)
  for begin in range(0,N,128):x[begin:begin+128]=tr(np.array(raw[begin:min(begin+128,N)]),channel_names=tuple(stats['channel_names']))
  encoder=build_encoder(a['author_source'],a['vendor'],seed=1729);state=torch.load(check(old['full_EEG']['model']),weights_only=True,map_location='cpu');assert state['role']=='full' and state['epoch']==50;encoder.load_state_dict(state['model'],strict=True);del state
  p=EEGNativeClassPrior(encoder).cuda().eval();initial=digest(p.state_dict());optimizer=torch.optim.AdamW([v for v in p.parameters() if v.requires_grad],lr=1e-4,weight_decay=.01)
  from models.labram import get_input_chans
  channels=get_input_chans([v.upper() for v in stats['channel_names']]);epoch0=0;updates=0;events=[];coverage=[];torch.cuda.reset_peak_memory_stats()
  if mode=='fit':
   profile=json.loads(check(s['profile_completion']).read_bytes());assert profile['status']=='PASS_REAL_NATIVE_CATEGORICAL_64_QUERY_THREE_UPDATE_PROFILE'
   commit=root/'commit.json'
   if commit.is_file():
    record=json.loads(commit.read_bytes());state=torch.load(check(record['checkpoint']),weights_only=False,map_location='cpu');assert state['spec_sha256']==sha(sp)
    p.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer']);torch.set_rng_state(state['CPU_RNG']);torch.cuda.set_rng_state_all(state['CUDA_RNG']);epoch0=state['epoch'];updates=state['updates'];events=state['events'];coverage=state['coverage'];del state
  loop=time.monotonic();phase='EXACT_NATIVE_CLASS_EEG_FIT';body_gradient=False
  for epoch in range(epoch0,3 if mode=='profile' else 10):
   order=torch.randperm(N,generator=torch.Generator().manual_seed(1729+epoch));seen=[];epoch_start=time.monotonic()
   for begin in range(0,N,64):
    guard();rows=order[begin:begin+64].numpy();eeg=torch.from_numpy(x[rows]).cuda();native_ll=torch.from_numpy(np.array(ll[rows])).cuda();optimizer.zero_grad(set_to_none=True)
    logits=p(eeg,input_chans=channels);loss,parts=conditional_loss(logits,native_ll,s['arm']);loss.backward()
    norm=sum(float(v.grad.square().sum()) for name,v in p.encoder.named_parameters() if v.grad is not None and not name.startswith(('head.','logit_scale')))**.5;body_gradient=body_gradient or norm>0
    gradient=float(torch.nn.utils.clip_grad_norm_(p.parameters(),1.));assert torch.isfinite(loss);optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1;seen.extend(rows.tolist())
    posterior=parts['posterior'];events.append(dict(epoch=epoch+1,batch=begin//64+1,queries=len(rows),optimized_loss=float(loss.detach()),native_coarse_NLL_nats_per_query=float(parts['native_coarse_NLL_nats_per_query'].mean()),posterior_entropy=float(-(posterior*posterior.clamp_min(1e-30).log()).sum(-1).mean()),EEG_class_entropy=float(-(logits.detach().softmax(-1)*logits.detach().log_softmax(-1)).sum(-1).mean()),preclip_gradient=gradient,backbone_gradient=norm))
    del eeg,native_ll,logits,loss,parts,posterior
   assert seen==order.tolist() and sorted(seen)==list(range(N));coverage.append(dict(epoch=epoch+1,queries=N,order_sha256=hashlib.sha256(order.numpy().tobytes()).hexdigest(),seconds=time.monotonic()-epoch_start));epochs=epoch+1;torch.cuda.synchronize();guard()
   if mode=='fit':
    phase='ATOMIC_WHOLE_EPOCH_CHECKPOINT';ck=root/('state_A.pt' if epochs%2 else 'state_B.pt');pending=ck.with_name(ck.name+'.pending');assert not pending.exists()
    torch.save(dict(spec_sha256=sha(sp),epoch=epochs,updates=updates,model=p.state_dict(),optimizer=optimizer.state_dict(),CPU_RNG=torch.get_rng_state(),CUDA_RNG=torch.cuda.get_rng_state_all(),events=events,coverage=coverage),pending);os.replace(pending,ck)
    record=dict(status='WHOLE_EPOCH_COMMITTED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),epoch=epochs,updates=updates,query_exposures=epochs*N,checkpoint=ref(ck),spec_sha256=sha(sp),last=events[-1],coverage=coverage)
    write(root/'commit.json',record,replace=(root/'commit.json').exists());write(root/'events.json',events,replace=(root/'events.json').exists());write(root/'progress.json',record,replace=(root/'progress.json').exists());phase='EXACT_NATIVE_CLASS_EEG_FIT'
    if time.monotonic()-started>600:break
  torch.cuda.synchronize();assert body_gradient;C.verify_protected_files(a);guard()
  result=dict(status='PASS_REAL_NATIVE_CATEGORICAL_64_QUERY_THREE_UPDATE_PROFILE' if mode=='profile' else 'COMPLETE_FIXED10_NATIVE_CATEGORICAL_FULL_T_FIT' if epochs==10 else 'WHOLE_EPOCH_NATIVE_CATEGORICAL_SEGMENT_CLOSED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=s['host'],arm=s['arm'],spec=ref(sp),epochs=epochs,updates=updates,EEG_queries_per_epoch=N,coverage=coverage,initial_EEG_model_digest=initial,EEG_parameters=sum(v.numel() for v in p.parameters()),backbone_gradient_nonzero=True,events=events if mode=='profile' else events[-5:],loop_seconds=time.monotonic()-loop,seconds=time.monotonic()-started,peak_GPU_allocated=torch.cuda.max_memory_allocated(),extra_CPU_peak=peak()-base,generator_updates=0,new_images=0)
  if mode=='profile':del p,optimizer;result['temporary_updates_discarded']=True
  elif epochs==10:
   final=root/'EEG_native_class_final.pt';assert not final.exists();torch.save(dict(model=p.state_dict(),epoch=10,updates=2070,spec_sha256=sha(sp),arm=s['arm']),final);result['final_model']=ref(final);result['checkpoint']=json.loads((root/'commit.json').read_bytes())['checkpoint']
  write(out/'completion.json',result)
  if mode=='fit' and epochs==10:write(root/'completion.json',result)
 except BaseException as e:write(out/'failure.json',dict(status='NATIVE_CATEGORICAL_FAILURE_PRESERVED_NO_RETRY',phase=phase,error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-started));raise
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);ap.add_argument('--job-name',required=True);ap.add_argument('--mode',choices=['profile','fit'],required=True);v=ap.parse_args();run(v.spec,v.job_name,v.mode)
