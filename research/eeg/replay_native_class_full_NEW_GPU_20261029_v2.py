"""Fixed native1000-class exact two-scale T likelihoods; no EEG/model fitting.

For a fixed first VQ code, all second-scale features are identical across
images. Share that context, retain the four true code targets separately.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import argparse,datetime,hashlib,json,shutil,subprocess,sys,time,traceback
from pathlib import Path
from audit_native_marginal_conditions_GPU_20261029_v1 import sha,ref,check,write,peak
def digest(state):
 import torch
 h=hashlib.sha256()
 for key,value in state.items():
  t=value.detach().cpu().contiguous();h.update(key.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
def run(sp,job):
 s=json.loads(Path(sp).read_bytes());assert s['status']=='REGISTERED_NATIVE_CONDITION_AUDIT_AND_CLASS_LIKELIHOOD_ENGINEERING_v1';host=s['host'];mode='full_T_NEW_replay';assert host=='new'
 out=Path(s['output_root'])/mode;out.mkdir();started=time.monotonic();base=peak();phase='EXCLUSIVE_FROZEN_ASSETS'
 try:
  for x in s['source_pins']:check(x)
  expected='run_gpu_job_v2.py' if host=='old' else 'run_BrainHIVE_two_host_GPU_job_20261028_v1.py'
  assert expected in (Path('/proc')/str(os.getppid())/'cmdline').read_bytes().decode().replace('\0',' ')
  assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==s['GPU_UUID']
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  r=json.loads((Path(s['repository_root'])/'audit'/(job+'_job.json')).read_bytes());assert r['pid']==os.getpid() and r['status']=='RUNNING' and float(r['max_seconds'] if host=='old' else r['maximum_seconds'])==900
  fs=json.loads(check(s['fit_spec']).read_bytes());sys.path.insert(0,fs['training_code']);import AR_anchored_training_contract_CPU_20261029_v1 as C
  a,_=C.load_admission(fs['training_admission']);C.validate_sources(a);C.verify_protected_files(a);sys.path.insert(0,a['vendor'])
  import numpy as np
  import torch
  from EEG_generative_backend_loader_20261029_v1 import build_and_load
  from EEG_native_marginal_core_20261029_v2 import NativeMarginalDecoder
  torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  def guard():
   assert time.monotonic()-started<800 and peak()-base<10<<30 and torch.cuda.max_memory_allocated()<20<<30
   assert shutil.disk_usage(out).free>=2<<30
  ids=json.loads(check(a['data']['EEG_identities']).read_bytes());tid=json.loads(check(a['data']['token_identities']).read_bytes());C.identity_join(ids['T'],tid)
  all_tokens=np.load(check(a['data']['tokens']),mmap_mode='r',allow_pickle=False);assert all_tokens.shape==(13230,680)
  positions=np.arange(13230);target=np.array(all_tokens[positions,:5],dtype=np.int64);contexts=np.unique(target[:,0])
  assert len(contexts)==755
  torch.cuda.reset_peak_memory_stats();vae,d,backend=build_and_load(a['author_source'],a['vendor'],check(a['public_VQ']),check(a['public_VAR']),seed=1729)
  state=torch.load(check(a['public_VAR']),weights_only=True,map_location='cpu',mmap=True);classes=state['class_emb.weight'].clone();del state
  before=digest({k:v for k,v in d.state_dict().items() if not k.startswith('embed_proj.')});vq_before=digest(vae.state_dict())
  native=NativeMarginalDecoder(d,classes).cuda().eval();vae.cuda().eval().requires_grad_(False);native_classes=classes[:1000].cuda()
  ll=np.lib.format.open_memmap(out/'log_likelihoods.npy',mode='w+',dtype=np.float32,shape=(len(positions),1000));ll[:]=np.nan;covered=np.zeros(len(positions),np.int8);context_seconds=[];phase='EXACT_NATIVE_CLASS_CONTEXT_GROUPED_LIKELIHOODS';loop=time.monotonic()
  with torch.inference_mode():
   for number,code in enumerate(contexts):
    guard();began=time.monotonic();rows=np.flatnonzero(target[:,0]==code);targets=torch.from_numpy(target[rows]).cuda()
    first=vae.quantize.embedding(torch.tensor([[int(code)]],device='cuda')).transpose(1,2).reshape(1,d.Cvae,1,1)
    _,following=vae.quantize.get_next_autoregressive_input(0,10,torch.zeros(1,d.Cvae,16,16,device='cuda'),first);features=following.flatten(2).transpose(1,2)
    for lo in range(0,1000,250):
     hi=min(lo+250,1000);logits=native.marginal_logits(features.expand(hi-lo,-1,-1),native_classes[lo:hi]);lp=logits.float().log_softmax(-1)
     values=lp[:,0,int(code),None].expand(-1,len(rows)).clone()
     for j in range(1,5):values+=lp[:,j,:].index_select(1,targets[:,j])
     ll[rows,lo:hi]=values.T.cpu().numpy();del logits,lp,values
    covered[rows]+=1;context_seconds.append(time.monotonic()-began)
    if number%25==0:
     ll.flush();p=out/'progress.json';value=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed_contexts=number+1,total_contexts=len(contexts),covered_T_rows=int((covered==1).sum()),loop_seconds=time.monotonic()-loop)
     pending=out/'progress.pending';pending.write_text(json.dumps(value)+'\n');os.replace(pending,p)
   assert (covered==1).all() and np.isfinite(ll).all() and (ll<=0).all();ll.flush();torch.cuda.synchronize();loop_seconds=time.monotonic()-loop
   # Independent direct complete680 forward on actual T examples/classes.
   errors=[]
   for index in (0,1):
    codes=torch.tensor(np.array(all_tokens[positions[index]]),device='cuda',dtype=torch.long)[None].expand(4,-1)
    full_features=vae.quantize.idxBl_to_var_input(list(torch.split(codes,[1,4,9,16,25,36,64,100,169,256],dim=1)))
    kk=[0,11,287,999];full=native(full_features,native_classes[kk]);direct=-torch.nn.functional.cross_entropy(full[:,:5].transpose(1,2),codes[:,:5],reduction='none').sum(-1)
    expected=torch.from_numpy(np.array(ll[index,kk])).cuda();error=float((direct-expected).abs().max());assert torch.allclose(direct,expected,atol=1e-4,rtol=2e-5);errors.append(error)
   assert digest(d.state_dict())==before and digest(vae.state_dict())==vq_before;C.verify_protected_files(a);guard()
  tensor=torch.from_numpy(np.array(ll));posterior=tensor.softmax(-1);entropy=-(posterior*posterior.clamp_min(1e-30).log()).sum(-1)
  np.save(out/'T_positions.npy',positions,allow_pickle=False)
  result=dict(status='COMPLETE_FIXED_NATIVE_CLASS_COARSE_LIKELIHOOD_BANK_NO_EEG_FIT',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),spec=ref(sp),host=host,scope=mode,T_rows=len(positions),native_condition_rows=list(range(1000)),native_null_class_excluded=True,first_scale_contexts=len(contexts),
   likelihoods=ref(out/'log_likelihoods.npy'),positions=ref(out/'T_positions.npy'),token_source=a['data']['tokens'],identities=a['data']['token_identities'],public_VAR=a['public_VAR'],public_VQ=a['public_VQ'],backend=backend,
   full680_vs_grouped5_max_abs_errors=errors,all_native_VAR_VQ_weights_unchanged=True,
   visual_only_uniform_prior_posterior_entropy_mean=float(entropy.mean()),visual_only_posterior_KL_to_uniform_mean=float(np.log(1000)-entropy.mean()),visual_only_posterior_max_probability_mean=float(posterior.max(-1).values.mean()),
   loop_seconds=loop_seconds,seconds=time.monotonic()-started,context_seconds_mean=float(np.mean(context_seconds)),peak_GPU_allocated=torch.cuda.max_memory_allocated(),extra_CPU_peak=peak()-base,
   EEG_models_fitted=0,query_images_generated=0,Dfit_Dcheck_or_V_effects_read=False,novelty_or_EEG_method_gain_established=False)
  write(out/'completion.json',result)
 except BaseException as e:write(out/'failure.json',dict(status='NATIVE_CLASS_LIKELIHOOD_BANK_FAILURE_PRESERVED',phase=phase,error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-started));raise
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);ap.add_argument('--job-name',required=True);v=ap.parse_args();run(v.spec,v.job_name)
