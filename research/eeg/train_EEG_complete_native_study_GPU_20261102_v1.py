"""All six EEG targets: real profile or fixed full-T five-epoch transactions."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import argparse,datetime,hashlib,json,resource,subprocess,time,traceback
from pathlib import Path
import EEG_complete_native_study_contract_CPU_20261102_v1 as C

def run(protocol,arm,phase,job):
 q=C.setup(protocol,formal=phase!='profile');assert arm in C.ARMS and phase in ('profile','fit5','fit10');C.resources(q,starting=True)
 base,a,image,stats=C.source_inputs(q,binary=False);repo=Path(q['repository_root']);r=C.read(repo/'audit'/(job+'_job.json'));assert r['status']=='RUNNING' and r['pid']==os.getpid() and r['max_seconds']==900
 assert 'run_gpu_job_v2.py' in Path('/proc',str(os.getppid()),'cmdline').read_bytes().decode().replace('\0',' ')
 assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==C.GPU
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 work=C.OUT/'tasks'/(arm+'_'+phase);work.mkdir(parents=True);started=time.monotonic();base_RSS=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
 try:
  C.roots();import numpy as np,torch
  import EEG_complete_native_runtime_20261102_v1 as R
  from EEG_complete_native_bank_reader_CPU_20261102_v1 import CompleteNativeBank
  from EEG_complete_native_study_core_20261102_v1 import training_loss
  torch.set_num_threads(2);torch.manual_seed(1729);torch.cuda.manual_seed_all(1729);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  x=R.inputs(q);bank=CompleteNativeBank(C.BANK);p=R.model(q,arm,'cuda');channels=R.channels(q)
  initial=R.digest(p.state_dict());optimizer=torch.optim.AdamW([v for v in p.parameters() if v.requires_grad],lr=1e-4,weight_decay=.01,betas=(.9,.999),eps=1e-8)
  root=C.OUT/'fits'/arm
  if phase!='profile':root.mkdir(parents=True,exist_ok=True)
  events=[];coverage=[];updates=0;epoch0=0;last=0.;body_nonzero=False;torch.cuda.reset_peak_memory_stats()
  def guard():
   nonlocal last
   assert time.monotonic()-started<q['GPU_soft_seconds'] and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-base_RSS<q['maximum_extra_CPU_bytes'] and torch.cuda.max_memory_allocated()<q['maximum_GPU_bytes']
   if time.monotonic()-last>10:C.resources(q);last=time.monotonic()
  def state(epoch):
   return dict(protocol=C.ref(protocol),arm=arm,purpose='ENGINEERING_ONLY_NEVER_GENERATION' if phase=='profile' else 'FORMAL_FIXED_FULL_T',epoch=epoch,updates=updates,query_exposures=updates*64 if phase=='profile' else epoch*13230,model=R.cpu_tree(p.state_dict()),optimizer=R.cpu_tree(optimizer.state_dict()),parameter_names=[name for name,v in p.named_parameters() if v.requires_grad],CPU_RNG=torch.get_rng_state(),CUDA_RNG=torch.cuda.get_rng_state_all(),events=events,coverage=coverage,initial_model_digest=initial,warm_source=base['full_EEG']['model'])
  def restore(s):
   assert s['protocol']==C.ref(protocol) and s['arm']==arm
   p.load_state_dict(s['model'],strict=True);optimizer.load_state_dict(s['optimizer']);torch.set_rng_state(s['CPU_RNG']);torch.cuda.set_rng_state_all(s['CUDA_RNG'])
  def one(rows,epoch,batch,capture=None):
   nonlocal updates,body_nonzero
   guard();eeg=torch.from_numpy(np.array(x[rows])).cuda();emission=torch.from_numpy(bank.rows(rows)).cuda();optimizer.zero_grad(set_to_none=True)
   logits,stay=p(eeg,input_chans=channels)
   if capture is not None:
    logits.retain_grad()
    if stay.requires_grad:stay.retain_grad()
   loss,extra=training_loss(logits,stay,emission,arm);assert torch.isfinite(loss);loss.backward()
   body=sum(float(v.grad.square().sum()) for name,v in p.named_parameters() if v.grad is not None and name.startswith('encoder.') and not name.startswith('encoder.head.'))**.5
   body_nonzero=body_nonzero or body>0
   if capture is not None:
    np.savez_compressed(work/f'gradient_trace_{capture}.npz',rows=rows,emissions=emission.cpu().numpy(),logits=logits.detach().cpu().numpy(),stay=stay.detach().cpu().numpy(),gradient_logits=logits.grad.cpu().numpy(),gradient_stay=np.zeros((len(rows),9),np.float32) if stay.grad is None else stay.grad.cpu().numpy(),loss=np.array(float(loss.detach()),np.float64))
   gradient=float(torch.nn.utils.clip_grad_norm_(p.parameters(),1.));assert np.isfinite(gradient);optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
   return dict(epoch=epoch,batch=batch,queries=len(rows),rows_sha256=hashlib.sha256(rows.astype(np.int64).tobytes()).hexdigest(),optimized_loss=float(loss.detach()),observed_NLL_nats_per_query=float(extra['observed_negative_log_evidence_nats'].mean()),objective=extra['optimized_objective'],optimized_scales=extra['optimized_scales'],preclip_parameter_gradient=gradient,backbone_gradient=body)
  if phase=='fit10':
   commit=C.read(root/'commit.json');assert commit['epoch']==5 and commit['updates']==1035
   prior_audit=C.read(root/'fit5_CPU_audit.json');assert prior_audit['status']=='PASS_ALL_FIXED_EEG_EPOCH_STATE_ADAM_COVERAGE' and prior_audit['checkpoint']==commit['checkpoint'] and prior_audit['protocol']==C.ref(protocol)
   s=torch.load(C.checked(commit['checkpoint']),map_location='cpu',weights_only=False);restore(s);events=s['events'];coverage=s['coverage'];updates=s['updates'];epoch0=5;assert initial==s['initial_model_digest'];del s
  elif phase=='fit5':assert not (root/'commit.json').exists()
  loop=time.monotonic()
  if phase=='profile':
   rows=np.linspace(0,13229,64,dtype=np.int64);before2=None
   for k in range(3):
    events.append(one(rows,0,k+1,capture=k))
    if k==1:before2=R.save_state(work/'before_resume_step2.pt',state(0))
   continuous=state(0);assert updates==3
   restored=torch.load(C.checked(before2),map_location='cpu',weights_only=False);restore(restored);updates=2
   replay=one(rows,0,3);assert updates==3
   assert R.tree_equal(p.state_dict(),continuous['model']) and R.tree_equal(optimizer.state_dict(),continuous['optimizer'])
   assert torch.equal(torch.get_rng_state(),continuous['CPU_RNG']) and R.tree_equal(torch.cuda.get_rng_state_all(),continuous['CUDA_RNG'])
   final=R.save_state(work/'after_profile_step3.pt',state(0));assert body_nonzero or arm=='image_prior_full'
   result=dict(status='PASS_REAL64_ALL_TARGET_PROFILE_AND_EXACT_STEP2_TO3_RESUME',protocol=C.ref(protocol),arm=arm,retained_trajectory_updates=3,actual_temporary_optimizer_calls=4,resume_replayed_updates=1,temporary_updates_discarded_from_formal_fit=True,temporary_states_not_generation_models=True,initial_digest=initial,final_digest=R.digest(p.state_dict()),before2=before2,after3=final,traces=[C.ref(work/f'gradient_trace_{k}.npz') for k in range(3)],events=events,parameter_count=sum(v.numel() for v in p.parameters()),backbone_gradient_after_initialization=body_nonzero,model_Adam_RNG_resume_bit_exact=True)
  else:
   stop=5 if phase=='fit5' else 10
   for epoch in range(epoch0,stop):
    order=torch.randperm(13230,generator=torch.Generator().manual_seed(1729+epoch)).numpy();seen=[]
    for lo in range(0,13230,64):
     rows=order[lo:lo+64];events.append(one(rows,epoch+1,lo//64+1));seen.extend(rows.tolist())
    assert seen==order.tolist() and sorted(seen)==list(range(13230));coverage.append(dict(epoch=epoch+1,queries=13230,order_sha256=hashlib.sha256(order.astype(np.int64).tobytes()).hexdigest()))
    snapshot=state(epoch+1);checkpoint=R.save_state(root/('state_A.pt' if (epoch+1)%2 else 'state_B.pt'),snapshot)
    C.write(root/'events.json',events,update=True);C.write(root/'commit.json',dict(status='COMPLETE_WHOLE_EPOCH_TRANSACTION',protocol=C.ref(protocol),arm=arm,epoch=epoch+1,updates=updates,query_exposures=(epoch+1)*13230,checkpoint=checkpoint,events=C.ref(root/'events.json')),update=True)
   if stop==5:
    assert not (root/'resume_half5.pt').exists();checkpoint=R.save_state(root/'resume_half5.pt',state(5))
    C.write(root/'commit.json',dict(status='COMPLETE_WHOLE_EPOCH_TRANSACTION',protocol=C.ref(protocol),arm=arm,epoch=5,updates=updates,query_exposures=66150,checkpoint=checkpoint,events=C.ref(root/'events.json')),update=True)
   assert updates==stop*207 and len(events)==updates and (body_nonzero or arm=='image_prior_full')
   result=dict(status='COMPLETE_FIXED10_FULL_T_EEG_ARM' if stop==10 else 'COMPLETE_FIXED_FIRST5_FULL_T_EEG_EPOCHS',protocol=C.ref(protocol),arm=arm,epoch=stop,updates=updates,query_exposures=stop*13230,coverage=coverage,checkpoint=checkpoint,events=C.ref(root/'events.json'),initial_digest=initial,final_digest=R.digest(p.state_dict()),warm_source=base['full_EEG']['model'],parameter_count=sum(v.numel() for v in p.parameters()),backbone_gradient_after_initialization=body_nonzero)
   if stop==10:
    path=root/'model_final.pt';assert not path.exists();result['final_model']=R.save_state(path,dict(protocol=C.ref(protocol),arm=arm,purpose='FORMAL_FIXED_FULL_T_FINAL_ONLY',epoch=10,updates=2070,query_exposures=132300,model=R.cpu_tree(p.state_dict()),warm_source=base['full_EEG']['model']));C.write(root/'completion.json',result)
  guard();C.checked(base['full_EEG']['model']);C.checked(base['full_EEG']['normalization'])
  result.update(seconds=time.monotonic()-started,loop_seconds=time.monotonic()-loop,extra_CPU_peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-base_RSS,GPU_peak=torch.cuda.max_memory_allocated(),new_research_images=0,frozen_generator_not_loaded_or_optimized=True)
  C.write(work/'completion.json',result)
 except BaseException as e:C.write(work/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),no_retry=True));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--arm',choices=C.ARMS,required=True);p.add_argument('--phase',choices=['profile','fit5','fit10'],required=True);p.add_argument('--job-name',required=True);v=p.parse_args();run(v.protocol,v.arm,v.phase,v.job_name)
