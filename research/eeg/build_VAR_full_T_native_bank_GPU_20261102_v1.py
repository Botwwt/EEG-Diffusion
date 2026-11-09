"""One complete64-row segment; every class/token consumed, no EEG fit."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import argparse,datetime,hashlib,json,resource,subprocess,sys,time,traceback
from pathlib import Path
import VAR_full_T_native_bank_contract_CPU_20261102_v1 as C
def run(protocol,part,job):
 q=C.setup(protocol);c,n,prev,_=C.dependency(q);C.resources(q,starting=True);s=C.read(c.previous()[0].NATIVE['path']);parent=C.read(n.mean_contract().PARENT['path']);repo=Path(parent['repository_root'])
 receipt=C.read(repo/'audit'/(job+'_job.json'));assert receipt['status']=='RUNNING' and receipt['pid']==os.getpid() and receipt['max_seconds']==900
 assert 'run_gpu_job_v2.py' in Path('/proc',str(os.getppid()),'cmdline').read_bytes().decode().replace('\0',' ')
 assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==C.GPU
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 lo,hi=C.part_range(part);out=C.OUT/f'part_{part:03d}';out.mkdir();began=time.monotonic();base=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
 try:
  for root in reversed(s['native_module_roots']):sys.path.insert(0,root)
  sys.path.insert(0,s['assets']['vendor']);sys.path.insert(0,str(n.CODE));sys.path.insert(0,str(c.CODE))
  import numpy as np,torch
  from EEG_generative_backend_loader_20261029_v1 import build_and_load
  from VAR_declared_FP16_kernel_20261102_v1 import DeclaredFP16Native
  from profile_VAR_multiscale_native_GPU_20261101_v1 import digest
  for name,pin in s['native_module_bindings'].items():assert Path(sys.modules[name].__file__).resolve()==C.checked(pin).resolve()
  for pin in s['assets']['data'].values():C.checked(pin)
  tokens=np.load(s['assets']['data']['tokens']['path'],mmap_mode='r');assert tokens.shape==(13230,680)
  target=np.array(tokens[lo:hi],dtype=np.int64);ids=C.read(s['assets']['data']['token_identities']['path']);assert len(ids)==13230
  assert len({r['stimulus_id'] for r in ids})==13230 and len({r['concept_id'] for r in ids})==1323
  assert all(int(row['query_index'])==i for i,row in enumerate(ids))
  torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  a=s['assets'];vae,d,backend=build_and_load(a['author_source'],a['vendor'],C.checked(a['public_VQ']),C.checked(a['public_VAR']),seed=1729)
  state=torch.load(a['public_VAR']['path'],map_location='cpu',weights_only=True,mmap=True);native=DeclaredFP16Native(d,state['class_emb.weight'].clone()).eval().requires_grad_(False);del state
  before=dict(VAR=digest(native),VQ=digest(vae));native.cuda();vae.cuda().eval().requires_grad_(False);torch.cuda.reset_peak_memory_stats()
  ll=np.full((hi-lo,1000,10),np.nan,np.float64);coverage=np.zeros((hi-lo,1000),np.uint8);batch_records=[];reference_matches=[]
  ref=np.load(c.OUT/'profile/true_token_logprobs.npy',mmap_mode='r');refpos=prev['T_positions'];last=0.
  def guard():
   nonlocal last
   assert time.monotonic()-began<q['GPU_soft_seconds'] and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-base<q['maximum_extra_CPU_bytes'] and torch.cuda.max_memory_allocated()<q['maximum_GPU_bytes']
   if time.monotonic()-last>10:C.resources(q);last=time.monotonic()
  loop=time.monotonic()
  with torch.inference_mode():
   for i,row in enumerate(range(lo,hi)):
    true=torch.from_numpy(target[i:i+1]).cuda();features=vae.quantize.idxBl_to_var_input(list(true.split([p*p for p in d.patch_nums],dim=1)))
    for a in range(0,1000,64):
     guard();b=min(a+64,1000);logits=native(features.expand(b-a,-1,-1),native.native_class_embedding[a:b]);lp=logits.float().log_softmax(-1).gather(-1,true.expand(b-a,-1)[:,:,None]).squeeze(-1)
     value=lp.cpu().numpy();assert value.dtype==np.float32 and np.isfinite(value).all() and (value<=0).all()
     direct=torch.stack([lp[:,x:y].double().sum(-1) for x,y in d.begin_ends],1).cpu().numpy();independent=[]
     for x,y in d.begin_ends:
      acc=np.zeros(b-a,np.float64)
      for token in range(x,y):acc+=value[:,token]
      independent.append(acc)
     cpu=np.stack(independent,1);assert np.array_equal(cpu,direct);ll[i,a:b]=cpu;coverage[i,a:b]+=1
     batch_records.append(dict(row=row,classes=[a,b],terms=int(value.size),streamed_FP32_token_logprobs_SHA256=hashlib.sha256(value.tobytes()).hexdigest(),all_token_GPU_vs_CPU_FP64_sums_exact=True))
     if row in refpos:
      err=float(np.abs(value-ref[refpos.index(row),a:b]).max());assert err<=1e-3;reference_matches.append(dict(row=row,classes=[a,b],maximum_same_kernel_reference_error=err))
     del logits,lp,value
    C.write(out/'progress.json',dict(completed_rows=i+1,total_rows=hi-lo,last_T_position=row,loop_seconds=time.monotonic()-loop),update=True)
   loop_seconds=time.monotonic()-loop;assert np.isfinite(ll).all() and np.array_equal(coverage,np.ones_like(coverage));assert before==dict(VAR=digest(native),VQ=digest(vae));guard()
  np.save(out/'scale_log_likelihoods.npy',ll,allow_pickle=False);np.save(out/'class_coverage.npy',coverage,allow_pickle=False);np.save(out/'T_positions.npy',np.arange(lo,hi,dtype=np.int64),allow_pickle=False)
  C.write(out/'completion.json',dict(status='COMPLETE_FULL680_ALL1000_NATIVE_T_BANK_SEGMENT',protocol=C.ref(protocol),part=part,T_range=[lo,hi],rows=hi-lo,classes=1000,scales=10,tokens=680,consumed_token_terms=(hi-lo)*1000*680,all_GPU_vs_streaming_CPU_scale_sums_exact=True,raw_all_token_arrays_persisted=False,streamed_FP32_token_hashes=batch_records,full_reference_T8_matches=reference_matches,source_tokens=s['assets']['data']['tokens'],source_identities=s['assets']['data']['token_identities'],VAR_VQ_state_digests=before,all_pretrained_weights_unchanged=True,backend=backend,precision=q['precision'],files=[C.ref(out/name) for name in ('scale_log_likelihoods.npy','class_coverage.npy','T_positions.npy')],loop_seconds=loop_seconds,seconds=time.monotonic()-began,GPU_peak=torch.cuda.max_memory_allocated(),extra_CPU_peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-base,new_training_updates=0,new_research_images=0))
 except BaseException as e:C.write(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),no_retry=True));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--part',type=int,required=True);p.add_argument('--job-name',required=True);v=p.parse_args();run(v.protocol,v.part,v.job_name)
