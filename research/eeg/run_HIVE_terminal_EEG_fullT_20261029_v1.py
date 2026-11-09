"""Whole sub-01 T, two fixed EEG objectives, original ordinary prefix frozen.

Separate finite bank/train parts; no Dcheck values, checkpoint selection, or
official fitting. Existing native EEG weights are fine-tuned, no new adapter.
"""
import argparse,datetime,gc,hashlib,json,math,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
def read(p):return json.loads(Path(p).read_bytes())
def ref(p):
 p=Path(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
def checked(x):assert ref(x['path'])==x;return Path(x['path'])
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def atomic_json(p,x):
 p=Path(p);temp=p.with_name(p.name+'.pending');temp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(temp,p)
def memory():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
def run(path,mode,part,arm):
 s=read(path);assert s['status']=='REGISTERED_FULL_T_NATIVE_TERMINAL_EEG_MATCHED_STATES' and s['host']=='old'
 for x in s['source_pins']+s['payloads']:checked(x)
 assert 'run_gpu_job_v2.py' in (Path('/proc')/str(os.getppid())/'cmdline').read_text()
 assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==s['GPU_UUID']
 root=Path(s['output_root']);folder=root/(f'bank_part_{part:02d}' if mode=='bank' else f'{arm}_part_{part:02d}');folder.mkdir();start=time.monotonic();base=memory();phase='RUNTIME'
 sys.path.insert(0,s['repository_root']);import brainhive_paired_study_contract_20261025_v1 as C
 C.setup_runtime(dict(runtime_manifest=s['runtime_manifest']));import brainhive_native_models_v1 as N
 import torch,numpy as np
 from torch.utils.checkpoint import checkpoint
 from diffusers import EulerAncestralDiscreteScheduler
 from brainhive_bounded_component_v1 import BoundedConditionAdapter
 from brainhive_component_generator_v1 import load_native,denoise
 from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
 torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.cuda.set_per_process_memory_fraction(min(1.,float(20<<30)/torch.cuda.get_device_properties(0).total_memory))
 def budget():assert time.monotonic()-start<800 and memory()-base<10<<30 and torch.cuda.max_memory_allocated()<20<<30 and shutil.disk_usage(root).free>3<<30
 def digest(module):
  h=hashlib.sha256()
  for n,v in module.state_dict().items():h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
  return h.hexdigest()
 try:
  phase='EXACT_NATIVE_LOAD';(folder/'loader').mkdir();observer=ConvertedWeightAudit(torch,folder/'loader');observer.install()
  try:pipe,A,P,prompts,host=load_native(training=False)
  finally:observer.restore()
  loading=observer.complete();ordinary=BoundedConditionAdapter(predict_risk=False).cuda().eval().requires_grad_(False);ostate=torch.load(s['ordinary']['path'],weights_only=True,map_location='cpu');assert ostate['calibration']['selected_strength']==1.;ordinary.load_state_dict(ostate['model'],strict=True)
  init=torch.load(s['parent']['path'],weights_only=True,map_location='cpu')['model'];scope=N.classes();model=scope['BrainMLP'](4250,1024,num_layers=1,dropout=.1).cuda().eval();model.load_state_dict(init,strict=True)
  gamma=init['block.layers.0.norm.weight'].cuda();beta=init['block.layers.0.norm.bias'].cuda();raw=torch.load(s['EEG']['path'],weights_only=True,map_location='cpu',mmap=True);assert raw.shape==(16540,63,250);rows=s['training_T_array_rows'];assert len(rows)==13230
  frozen=dict(ordinary=ordinary,A=A,P=P,UNet=pipe.unet,VAE=pipe.vae);before={k:digest(v) for k,v in frozen.items()};initial_model=digest(model)
  def eeg(indices):return raw[[rows[int(i)] for i in indices]][:,46:63,:].contiguous().cuda()
  def conditioned(x):
   c=model(x);return ordinary(c,gamma[None].expand_as(c),beta[None].expand_as(c))[0]
  if mode=='bank':
   phase='NATIVE_ORDINARY_FULL_T_PREFIX_BANK';assert 0<=part<s['bank_parts'];model.requires_grad_(False);lo=part*s['bank_rows_per_part'];hi=min(13230,lo+s['bank_rows_per_part']);dest=folder/'prefix.npy';array=np.lib.format.open_memmap(str(dest)+'.pending',mode='w+',dtype=np.float16,shape=(hi-lo,4,64,64));batch_proofs=[];tick=time.monotonic()
   for begin in range(lo,hi,8):
    end=min(begin+8,hi);budget();indices=list(range(begin,end));actual_rows=rows[begin:end];seeds=[int(hashlib.sha256(('HIVE_terminal_fullT_prefix_'+str(row)).encode()).hexdigest()[:8],16) for row in actual_rows];states={}
    def callback(p,i,t,values):
     if i==2:states['prefix']=values['latents'].detach().clone()
     return values
    with torch.no_grad():tokens=P(A(conditioned(eeg(indices)).half()));pipe(height=512,width=512,generator=[torch.Generator(device='cuda').manual_seed(seed) for seed in seeds],prompt_embeds=prompts[0].expand(end-begin,-1,-1),pooled_prompt_embeds=prompts[1].expand(end-begin,-1),cross_attention_kwargs={'ip_hidden_states':tokens},num_inference_steps=4,guidance_scale=0.,output_type='latent',callback_on_step_end=callback,callback_on_step_end_tensor_inputs=['latents'])
    value=states['prefix'].cpu().numpy();assert value.shape==(end-begin,4,64,64) and np.isfinite(value).all();array[begin-lo:end-lo]=value;batch_proofs.append(dict(start=begin,stop=end,array_rows=actual_rows,seeds=seeds,prefix_tensor_sha256=hashlib.sha256(value.tobytes()).hexdigest(),native_NFE=4,actual_generation_batch=end-begin))
    if (begin-lo)%128==0:atomic_json(folder/'progress.json',dict(mode=mode,part=part,committed_queries=end-lo,expected=hi-lo,seconds=time.monotonic()-start))
   array.flush();del array;os.replace(str(dest)+'.pending',dest);torch.cuda.synchronize();assert initial_model==digest(model) and before=={k:digest(v) for k,v in frozen.items()};budget();write(folder/'completion.json',dict(status='COMPLETE_NATIVE_FROZEN_ORDINARY_T_PREFIX_PART',part=part,start=lo,stop=hi,queries=hi-lo,prefix=ref(dest),batches=batch_proofs,generation_seconds=time.monotonic()-tick,native_prefix_batching_declared=True,not_claimed_pixel_identical_to_single_query=True,all_frozen_weights_unchanged=True,protocol=ref(path),seconds=time.monotonic()-start,peak_GPU_allocated=torch.cuda.max_memory_allocated(),extra_CPU_peak=memory()-base));return
  assert mode=='train' and arm in ('real_prefix','Gaussian');assert 0<=part<s['training_parts_per_arm'];phase='BIND_COMPLETE_T_PREFIX_AND_TARGETS'
  banks=[]
  for j in range(s['bank_parts']):
   done=read(root/f'bank_part_{j:02d}/completion.json');assert done['start']==j*s['bank_rows_per_part'] and done['stop']==min(13230,(j+1)*s['bank_rows_per_part']);p=checked(done['prefix']);banks.append(np.load(p,mmap_mode='r'))
  targets=torch.load(s['F_targets']['path'],weights_only=True,map_location='cpu',mmap=True);latent=np.load(s['latent_bank']['path'],mmap_mode='r');assert latent.shape==(13230,4,64,64)
  for name,p in model.named_parameters():p.requires_grad_(not name.startswith('block.layers.0.norm.'))
  assert sum(p.numel() for p in model.parameters() if p.requires_grad)==6450176
  opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01);rng=torch.Generator(device='cuda').manual_seed(1730);order=torch.randperm(13230,generator=torch.Generator(device='cpu').manual_seed(1731));order_sha=hashlib.sha256(order.numpy().tobytes()).hexdigest();assert order_sha==s['training_order_sha256']
  updates=0;exposures=0;coverage=torch.zeros(13230,dtype=torch.bool);log_prefix=[];init_sha=initial_model
  if part:
   previous=read(root/f'{arm}_part_{part-1:02d}/completion.json');saved=torch.load(checked(previous['checkpoint']),weights_only=False,map_location='cpu');assert saved['protocol_sha256']==ref(path)['sha256'] and saved['arm']==arm and saved['order_sha256']==order_sha;model.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['optimizer']);rng.set_state(saved['noise_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);updates=saved['updates'];exposures=saved['exposures'];coverage=saved['coverage'];log_prefix=saved['log_prefix'];assert saved['initial_model_sha256']==init_sha
  lower=part*s['updates_per_part'];upper=min(s['updates'],lower+s['updates_per_part']);assert updates==lower and exposures==min(lower*8,13230) and int(coverage.sum())==exposures
  cfg=read(s['scheduler_config']['path']);sigma=s['terminal_sigma'];events=[];tick=time.monotonic();loss_sum=0.;endpoint_sum=0.;token_sum=0.;count=0
  phase='ONE_COMPLETE_T_PASS_MATCHED_TERMINAL_OBJECTIVE'
  def save_checkpoint(dest):
   state=dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},optimizer=opt.state_dict(),noise_rng=rng.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),coverage=coverage,updates=updates,exposures=exposures,protocol_sha256=ref(path)['sha256'],order_sha256=order_sha,initial_model_sha256=init_sha,arm=arm,log_prefix=log_prefix+events)
   pending=dest.with_name(dest.name+'.pending');torch.save(state,pending);os.replace(pending,dest)
  for step in range(lower,upper):
   budget();indices=order[step*8:min((step+1)*8,13230)].tolist();n=len(indices);assert n>0 and not coverage[indices].any();actual_rows=[rows[i] for i in indices];real=torch.from_numpy(np.array(latent[indices],copy=True)).cuda();noise=torch.randn(real.shape,generator=rng,device='cuda');z=(real+sigma*noise).half()
   if arm=='real_prefix':z=torch.from_numpy(np.stack([banks[i//s['bank_rows_per_part']][i%s['bank_rows_per_part']] for i in indices])).cuda()
   with torch.no_grad():teacher=P(A(targets[actual_rows].cuda().half())).detach()
   opt.zero_grad(set_to_none=True);corrected=conditioned(eeg(indices));tokens=P(A(corrected.half()));scheduler=EulerAncestralDiscreteScheduler.from_config(cfg);scheduler.set_timesteps(4,device='cuda');t=scheduler.timesteps[3];scaled=scheduler.scale_model_input(z,t);assert float(scheduler.sigmas[3])==sigma
   eps=checkpoint(lambda x,tt,tk:denoise(pipe,prompts,x,tt,tk),scaled,t,tokens,use_reentrant=False);endpoint=scheduler.step(eps,t,z,generator=torch.Generator(device='cuda').manual_seed(1729)).prev_sample;endpoint_loss=(endpoint.float()-real).square().mean();token_loss=(1-torch.nn.functional.cosine_similarity(tokens.float(),teacher.float(),dim=-1)).mean();loss=endpoint_loss+.1*token_loss;assert torch.isfinite(loss)
   (loss*1024.).backward()
   for p in model.parameters():
    if p.requires_grad:assert p.grad is not None;p.grad.div_(1024.);assert torch.isfinite(p.grad).all()
   norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.));opt.step();coverage[indices]=True;updates+=1;exposures+=n;loss_sum+=float(loss)*n;endpoint_sum+=float(endpoint_loss)*n;token_sum+=float(token_loss)*n;count+=n
   if updates%50==0 or updates==upper:
    torch.cuda.synchronize();event=dict(arm=arm,part=part,updates=updates,total_updates=s['updates'],exposures=exposures,interval_queries=count,mean_loss=loss_sum/count,mean_endpoint_MSE=endpoint_sum/count,mean_token_loss=token_sum/count,last_unclipped_gradient_norm=norm,seconds=time.monotonic()-tick,peak_GPU_allocated=torch.cuda.max_memory_allocated());events.append(event);print(json.dumps(event),flush=True);atomic_json(folder/'progress.json',event);loss_sum=endpoint_sum=token_sum=0.;count=0
   if updates%100==0:save_checkpoint(folder/'live_resume.pt')
  torch.cuda.synchronize();train_seconds=time.monotonic()-tick;dest=folder/'checkpoint.pt';save_checkpoint(dest);assert torch.equal(model.block.layers[0].norm.weight,gamma) and torch.equal(model.block.layers[0].norm.bias,beta);assert before=={k:digest(v) for k,v in frozen.items()};checked(s['ordinary']);budget()
  assert updates==upper and int(coverage.sum())==exposures
  if upper==s['updates']:
   assert updates==1654 and exposures==13230 and coverage.all();torch.save(dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},protocol_sha256=ref(path)['sha256'],arm=arm,updates=updates,exposures=exposures,output_LN_frozen=True,ordinary=s['ordinary'],normal_inference_target_access=False),folder/'EEG_final.pt')
  write(folder/'completion.json',dict(status='COMPLETE_MATCHED_NATIVE_TERMINAL_EEG_TRAIN_PART',arm=arm,part=part,updates=updates,exposures=exposures,order_sha256=order_sha,initial_model_sha256=init_sha,checkpoint=ref(dest),final_model=ref(folder/'EEG_final.pt') if (folder/'EEG_final.pt').is_file() else None,all_frozen_weights_unchanged=True,output_LN_affine_frozen=True,training_seconds=train_seconds,seconds=time.monotonic()-start,peak_GPU_allocated=torch.cuda.max_memory_allocated(),extra_CPU_peak=memory()-base,protocol=ref(path),events=events))
 except BaseException as error:write(folder/'failure.json',dict(phase=phase,arm=arm,part=part,error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-start));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--mode',choices=['bank','train'],required=True);p.add_argument('--part',type=int,required=True);p.add_argument('--arm',default='');a=p.parse_args();run(a.protocol,a.mode,a.part,a.arm)
