"""Ordinary training-seed reproducibility, same scientific recipe, separate outputs.

No method modification or innovation claim. Full10 real OOF-T, 3 passes,
epsilon MSE+.1 token angular error. Finite parts preserve exact Adam/RNG/order.
"""
import argparse,datetime,gc,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
def read(p):return json.loads(Path(p).read_bytes())
def ref(p):
 p=Path(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
def checked(x):
 got=ref(x['path']);assert got['sha256']==x['sha256']
 if 'bytes' in x:assert got['bytes']==x['bytes']
 return Path(x['path'])
def atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_name(p.name+'.pending');temp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(temp,p)
def memory():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
def seed(number,namespace):return int(hashlib.sha256((str(number)+'|BrainHIVE_bounded_v1|'+namespace).encode()).hexdigest()[:16],16)%(2**63-1)
def run(path,mode,number,part):
 s=read(path);assert s['host']=='old'
 if mode=='profile':assert s['status']=='REGISTERED_ORDINARY_SEED_REPRO_ENGINEERING_ONLY' and number==1729 and part==0
 else:assert s['status']=='REGISTERED_ORDINARY_SEED_REPRO_FULL_T' and number in [2027,3407] and 0<=part<42;checked(s['actual_engineering_completion'])
 for x in s['source_pins']+s['payloads']:checked(x)
 assert 'run_gpu_job_v2.py' in (Path('/proc')/str(os.getppid())/'cmdline').read_text()
 assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==s['GPU_UUID']
 out=Path(s['output_root'])/('profile' if mode=='profile' else f'seed{number}/part{part:02d}');out.mkdir(parents=True);started=time.monotonic();base=memory();phase='RUNTIME'
 sys.path.insert(0,s['repository_root']);import brainhive_paired_study_contract_20261025_v1 as C
 C.setup_runtime(dict(runtime_manifest=s['runtime_manifest']))
 import torch,numpy as np
 import torch.nn.functional as F
 from diffusers import DDPMScheduler
 from brainhive_bounded_component_v1 import BoundedConditionAdapter
 from brainhive_component_generator_v1 import load_native,denoise
 from brainhive_component_data_v1 import load_training
 from train_BrainHIVE_bounded_component_v1 import development
 from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.cuda.set_per_process_memory_fraction(min(1.,float(20<<30)/torch.cuda.get_device_properties(0).total_memory))
 def budget():assert time.monotonic()-started<820 and memory()-base<10<<30 and torch.cuda.max_memory_allocated()<20<<30 and shutil.disk_usage(out).free>10<<30
 def digest(model):
  h=hashlib.sha256()
  for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
  return h.hexdigest()
 def save(p,x):temp=p.with_name(p.name+'.pending');torch.save(x,temp);os.replace(temp,p)
 try:
  phase='NATIVE_FROZEN_HOST_AND_COMPLETE_TRUE_OOF';(out/'loader').mkdir();observer=ConvertedWeightAudit(torch,out/'loader');observer.install()
  try:pipe,A,P,prompts,host=load_native(training=True)
  finally:observer.restore()
  loader=observer.complete();frozen=dict(A=A,P=P,UNet=pipe.unet);before={k:digest(v) for k,v in frozen.items()}
  # VAE/text encoders have no role in cached-latent epsilon training. Release
  # only unused modules after exact host load and fixed prompt construction.
  pipe.vae=None;pipe.text_encoder=None;pipe.text_encoder_2=None;gc.collect();torch.cuda.empty_cache();inputs=load_training();assert inputs['input_contract_sha256']==s['input_contract']['sha256']
  receipt=read(s['latent_receipt']['path']);assert receipt['array_rows']==inputs['array_rows'];array=np.load(checked(s['latent_bank']),mmap_mode='r');assert array.shape==(13230,4,64,64) and array.dtype==np.float32;image_latents=torch.tensor(array,device='cuda');del array
  scheduler=DDPMScheduler.from_config(pipe.scheduler.config);assert scheduler.config.prediction_type=='epsilon'
  orders=[torch.randperm(132300,generator=torch.Generator().manual_seed(seed(number,'pass'+str(i)))) for i in range(3)];order_SHA=[hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in orders]
  if number==1729:assert order_SHA==read(s['ordinary_input_provenance']['path'])['orders_sha256']
  def fresh():
   torch.manual_seed(number);m=BoundedConditionAdapter(predict_risk=False).cuda();assert sum(p.numel() for p in m.parameters())==525568
   return m,torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=.01),torch.Generator(device='cuda').manual_seed(seed(number,'shared_training_noise'))
  def step(m,opt,rng,update):
   epoch,within=divmod(update,8269);offset=within*16;indices=orders[epoch][offset:min(offset+16,132300)];idx=indices.cuda();pos=(indices%13230).cuda();aid=inputs['affine_ids'][idx];c=inputs['condition'][idx];w=inputs['weight'][aid];b=inputs['bias'][aid];latent=image_latents[pos]
   noise=torch.randn(latent.shape,device='cuda',generator=rng);t=torch.randint(0,1000,(len(indices),),device='cuda',generator=rng);noisy=scheduler.add_noise(latent,noise,t);opt.zero_grad(set_to_none=True);changed,info=m(c,w,b)
   if update==0:assert torch.equal(changed,c)
   with torch.autocast('cuda',dtype=torch.bfloat16):
    with torch.no_grad():real=P(A(inputs['target'][pos]))
    tokens=P(A(changed));pred=denoise(pipe,prompts,noisy,t,tokens);epsilon_loss=F.mse_loss(pred.float(),noise);token_loss=(1-F.cosine_similarity(tokens.float(),real.float(),dim=-1)).mean();loss=epsilon_loss+.1*token_loss
   assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);assert m.direction.weight.grad is not None and all(p.grad is None for p in pipe.unet.parameters());opt.step()
   return dict(epoch=epoch,indices=indices,n=len(indices),loss=float(loss),epsilon=float(epsilon_loss),token=float(token_loss),gradient_norm=float(norm),timesteps=t.tolist(),direction_grad_norm=float(m.direction.weight.grad.norm()),trunk_grad_norm=float(m.trunk[0].weight.grad.norm()))
  def snapshot(m,opt,rng,updates,coverage,seconds,total_loss):return dict(model={k:v.detach().cpu().clone() for k,v in m.state_dict().items()},optimizer=opt.state_dict(),noise_rng=rng.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),updates=updates,seen=int(coverage.sum()),coverage=coverage,training_seconds=seconds,total_loss=total_loss,seed=number,protocol_sha256=ref(path)['sha256'],orders_sha256=order_SHA)
  def restore(m,opt,rng,x):m.load_state_dict(x['model'],strict=True);opt.load_state_dict(x['optimizer']);rng.set_state(x['noise_rng']);torch.set_rng_state(x['torch_rng']);torch.cuda.set_rng_state_all(x['cuda_rng'])
  def equal(a,b):
   if isinstance(a,torch.Tensor):return torch.equal(a.cpu(),b.cpu())
   if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
   if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
   return a==b
  if mode=='profile':
   phase='ACTUAL_FULL_INPUT_FOUR_STEP_EXACT_STATE_RESUME';gold,opt,rng=fresh();initial=digest(gold);proof=[];tick=time.monotonic();gold_coverage=torch.zeros(3,132300,dtype=torch.bool)
   for update in range(4):
    budget();x=step(gold,opt,rng,update);gold_coverage[x['epoch'],x['indices']]=True;proof.append(x)
   torch.cuda.synchronize();elapsed=time.monotonic()-tick;expected=snapshot(gold,opt,rng,4,gold_coverage,elapsed,0.);del gold,opt,rng;gc.collect();torch.cuda.empty_cache()
   model,opt,rng=fresh();assert digest(model)==initial;replay=[];replay_coverage=torch.zeros(3,132300,dtype=torch.bool)
   for update in range(2):
    budget();x=step(model,opt,rng,update);replay_coverage[x['epoch'],x['indices']]=True;replay.append(x)
   boundary=out/'temporary_resume_after2.pt';before_boundary=snapshot(model,opt,rng,2,replay_coverage,0.,0.);save(boundary,before_boundary);del model,opt,rng;gc.collect();torch.cuda.empty_cache();model,opt,rng=fresh();saved=torch.load(boundary,weights_only=False,map_location='cpu');restore(model,opt,rng,saved);replay_coverage=saved['coverage'];restored_boundary=snapshot(model,opt,rng,2,replay_coverage,0.,0.);boundary_exact={k:equal(restored_boundary[k],before_boundary[k]) for k in ('model','optimizer','noise_rng','torch_rng','cuda_rng','orders_sha256','coverage','seen')};assert all(boundary_exact.values());del saved
   for update in range(2,4):
    budget();x=step(model,opt,rng,update);replay_coverage[x['epoch'],x['indices']]=True;replay.append(x)
   got=snapshot(model,opt,rng,4,replay_coverage,0.,0.);save(out/'gold_uninterrupted4.pt',expected);save(out/'serialized_resume4.pt',got)
   split_exact={k:equal(got[k],expected[k]) for k in ('model','optimizer','noise_rng','torch_rng','cuda_rng','orders_sha256','coverage','seen')}
   def differences(x,y):
    result={}
    def walk(a,b,prefix):
     if isinstance(a,torch.Tensor):
      aa=a.detach().cpu().double();bb=b.detach().cpu().double();delta=aa-bb;result[prefix]=dict(exact=torch.equal(aa,bb),maximum_absolute=float(delta.abs().max()),relative_L2=float(delta.norm()/aa.norm().clamp_min(1e-30)))
     elif isinstance(a,dict):
      for k in a:walk(a[k],b[k],prefix+'/'+str(k))
     elif isinstance(a,(list,tuple)):
      for k,(v,w) in enumerate(zip(a,b)):walk(v,w,prefix+'/'+str(k))
    walk(x,y,'');return result
   split_differences=differences(got['model'],expected['model']);del model,opt,rng;gc.collect();torch.cuda.empty_cache();repeat_model,repeat_opt,repeat_rng=fresh();repeat_proof=[];repeat_coverage=torch.zeros(3,132300,dtype=torch.bool)
   for update in range(4):
    budget();x=step(repeat_model,repeat_opt,repeat_rng,update);repeat_coverage[x['epoch'],x['indices']]=True;repeat_proof.append(x)
   repeated=snapshot(repeat_model,repeat_opt,repeat_rng,4,repeat_coverage,0.,0.);save(out/'independent_uninterrupted_repeat4.pt',repeated)
   repeat_exact={k:equal(repeated[k],expected[k]) for k in ('model','optimizer','noise_rng','torch_rng','cuda_rng','orders_sha256','coverage','seen')}
   clean=lambda items:[{k:(v.tolist() if isinstance(v,torch.Tensor) else v) for k,v in item.items()} for item in items]
   diagnostic=dict(status='MEASURED_RESTORATION_AND_UNINTERRUPTED_KERNEL_REPEAT_SEPARATELY',serialized_boundary_exact=boundary_exact,split_future_exact=split_exact,uninterrupted_repeat_exact=repeat_exact,split_model_differences=split_differences,uninterrupted_model_differences=differences(repeated['model'],expected['model']),split_optimizer_differences=differences(got['optimizer'],expected['optimizer']),uninterrupted_optimizer_differences=differences(repeated['optimizer'],expected['optimizer']),gold_steps=clean(proof),split_steps=clean(replay),uninterrupted_repeat_steps=clean(repeat_proof),all_original_future_bit_exact_guards_retained=True,temporary_updates=12,new_formal_updates=0,all_frozen_weights_unchanged=before=={k:digest(v) for k,v in frozen.items()},peak_GPU_allocated=torch.cuda.max_memory_allocated(),extra_CPU_peak=memory()-base,seconds=time.monotonic()-started)
   atomic(out/'resume_numerical_diagnostic.json',diagnostic)
   assert all(split_exact.values()) and all(repeat_exact.values()),'Original future bit-exact criterion retained; diagnostic saved, no automatic admission'
   assert equal(proof,replay);assert proof[0]['direction_grad_norm']>0 and proof[3]['trunk_grad_norm']>0
   clean_proof=[{k:(v.tolist() if isinstance(v,torch.Tensor) else v) for k,v in item.items()} for item in proof]
   assert before=={k:digest(v) for k,v in frozen.items()};checked(s['ordinary']);budget();atomic(out/'completion.json',dict(status='PASS_ORDINARY_REPRO_FULL_INPUT_NATIVE_LOSS_EXACT_SERIALIZED_RESUME_ENGINEERING',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol=ref(path),queries_per_pass=132300,training_seed1729_original_orders_match=True,initial_head_sha256=initial,temporary_updates=12,temporary_query_exposures=192,temporary_unique_queries=64,temporary_checkpoint_retained_for_engineering_audit_only=True,temporary_weights_excluded_from_formal_fits=True,optimizer_model_all_RNG_and_loss_sequence_bit_exact=True,scientific_steps=clean_proof,four_uninterrupted_training_seconds=elapsed,estimated_24807_training_seconds=elapsed/4*24807,all_frozen_weights_unchanged=True,peak_GPU_allocated=torch.cuda.max_memory_allocated(),extra_CPU_peak=memory()-base,seconds=time.monotonic()-started,formal_fit_admitted=False,new_research_images=0,NEW_contacted=False,loader=loader));return
  phase='FIXED_SEED_FULL_T_THREE_PASSES';model,opt,rng=fresh();initial=digest(model);coverage=torch.zeros(3,132300,dtype=torch.bool);seconds=total_loss=0.;lower=part*600;upper=min(24807,lower+600)
  if part:
   previous=read(Path(s['output_root'])/f'seed{number}/part{part-1:02d}/completion.json');saved=torch.load(checked(previous['checkpoint']),weights_only=False,map_location='cpu');assert saved['updates']==lower and saved['seed']==number and saved['protocol_sha256']==ref(path)['sha256'] and saved['orders_sha256']==order_SHA;restore(model,opt,rng,saved);coverage=saved['coverage'];seconds=saved['training_seconds'];total_loss=saved['total_loss'];del saved
  expected_seen=(lower//8269)*132300+(lower%8269)*16;assert int(coverage.sum())==expected_seen;tick=time.monotonic();events=[]
  for update in range(lower,upper):
   budget();x=step(model,opt,rng,update);indices=x.pop('indices');epoch=x.pop('epoch');n=x.pop('n');assert not coverage[epoch,indices].any();coverage[epoch,indices]=True;total_loss+=x['loss']*n
   if (update+1)%64==0 or update+1==upper:
    ev=dict(seed=number,part=part,updates=update+1,expected_updates=24807,seen=int(coverage.sum()),pass_number=epoch+1,cumulative_mean_loss=total_loss/int(coverage.sum()),training_seconds=seconds+time.monotonic()-tick,**x);events.append(ev);atomic(out/'progress.json',ev);print(json.dumps(ev),flush=True)
   if (update+1)%128==0:save(out/'live_resume.pt',snapshot(model,opt,rng,update+1,coverage,seconds+time.monotonic()-tick,total_loss))
  torch.cuda.synchronize();seconds+=time.monotonic()-tick;dest=out/'checkpoint.pt';save(dest,snapshot(model,opt,rng,upper,coverage,seconds,total_loss));decoded=torch.load(dest,weights_only=False,map_location='cpu');assert decoded['updates']==upper and int(decoded['coverage'].sum())==int(coverage.sum());assert {int(v['step']) for v in decoded['optimizer']['state'].values()}=={upper};assert all(torch.equal(v.cpu(),decoded['model'][k]) for k,v in model.state_dict().items());del decoded
  assert before=={k:digest(v) for k,v in frozen.items()};checked(s['ordinary']);budget();result=dict(status='COMPLETE_EXACT_PARTIAL_ORDINARY_SEED_REPRO',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),seed=number,part=part,updates=upper,seen=int(coverage.sum()),checkpoint=ref(dest),initial_head_sha256=initial,orders_sha256=order_SHA,events=events,all_frozen_weights_unchanged=True,training_seconds=seconds,seconds=time.monotonic()-started,peak_GPU_allocated=torch.cuda.max_memory_allocated(),extra_CPU_peak=memory()-base,protocol=ref(path))
  if upper==24807:
   assert coverage.all() and int(coverage.sum())==396900;calibration=development(model.eval(),A,P,'ordinary');atomic(out/'calibration.json',calibration);final=out/'component_final.pt';save(final,dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},arm='ordinary',seed=number,updates=24807,seen_queries=396900,passes=3,calibration=calibration,protocol_sha256=ref(path)['sha256']));result.update(status='COMPLETE_FULL10_TRUE_OOF_T_THREE_PASSES_ORDINARY_REPLICATE',component=ref(final),calibration=ref(out/'calibration.json'),fixed_final_not_selected_by_images=True)
  atomic(out/'completion.json',result)
 except BaseException as error:atomic(out/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-started));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--mode',choices=['profile','train'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--part',type=int,required=True);a=p.parse_args();run(a.protocol,a.mode,a.seed,a.part)
