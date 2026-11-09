"""Real T engineering only: native last-step parity, gradient and cost profile.

Four temporary updates (two per matched state source) are discarded. Native
ordinary, priors and sampler remain frozen. No official fit or research images.
"""
import argparse,datetime,gc,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
def read(p):return json.loads(Path(p).read_bytes())
def ref(p):
 p=Path(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
def checked(x):assert ref(x['path'])==x;return Path(x['path'])
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def memory():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
def run(path):
 s=read(path);assert s['host']=='old' and s['status']=='REGISTERED_ENGINEERING_NATIVE_TERMINAL_EEG_PROFILE'
 for x in s['source_pins']+s['payloads']:checked(x)
 assert 'run_gpu_job_v2.py' in (Path('/proc')/str(os.getppid())/'cmdline').read_text()
 assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==s['GPU_UUID']
 root=Path(s['output_root']);root.mkdir();started=time.monotonic();base=memory();phase='RUNTIME'
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
 def budget():assert time.monotonic()-started<800 and memory()-base<10<<30 and torch.cuda.max_memory_allocated()<20<<30 and shutil.disk_usage(root).free>2<<30
 try:
  phase='ALL240_GPU_NATIVE_TERMINAL_REPLAY';cfg=read(s['scheduler_config']['path']);replay=[]
  for i in range(80):
   bank=np.load(Path(s['visual_probe_root'])/'terminal_tensors'/f'{i:03d}.npz',allow_pickle=False)
   for arm in ('ordinary','correct_visual_last','sameconcept_visual_last'):
    x=torch.from_numpy(bank[arm+'_prefix'].copy()).cuda();eps=torch.from_numpy(bank[arm+'_epsilon'].copy()).cuda();expected=torch.from_numpy(bank[arm+'_terminal'].copy()).cuda();sch=EulerAncestralDiscreteScheduler.from_config(cfg);sch.set_timesteps(4,device='cuda');t=sch.timesteps[3];sch.scale_model_input(x,t);got=sch.step(eps,t,x,generator=torch.Generator(device='cuda').manual_seed(1729)).prev_sample
    error=float((got.float()-expected.float()).abs().max());replay.append(dict(query=i,arm=arm,bit_exact=torch.equal(got,expected),maximum=error))
   budget()
  assert all(x['bit_exact'] for x in replay),'Saved native terminal transition must be exactly reproducible on its actual GPU backend'
  write(root/'all240_GPU_native_terminal_replay.json',dict(status='ALL240_NATIVE_GPU_TERMINAL_TRANSITIONS_BIT_EXACT',transitions=replay,CPU_strict_guard_failure_retained=True,new_UNet_calls=0))
  phase='EXACT_NATIVE_LOADING';(root/'loader').mkdir();observer=ConvertedWeightAudit(torch,root/'loader');observer.install()
  try:pipe,A,P,prompts,host=load_native(training=False)
  finally:observer.restore()
  loader=observer.complete();budget()
  def digest(module):
   h=hashlib.sha256()
   for n,v in module.state_dict().items():h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
   return h.hexdigest()
  ordinary=BoundedConditionAdapter(predict_risk=False).cuda().eval().requires_grad_(False);o=torch.load(s['ordinary']['path'],weights_only=True,map_location='cpu');assert o['calibration']['selected_strength']==1.;ordinary.load_state_dict(o['model'],strict=True)
  parent_state=torch.load(s['parent']['path'],weights_only=True,map_location='cpu');scope=N.classes();parent=scope['BrainMLP'](4250,1024,num_layers=1,dropout=.1).cuda().eval().requires_grad_(False);parent.load_state_dict(parent_state['model'],strict=True)
  frozen=dict(parent=parent,ordinary=ordinary,A=A,P=P,UNet=pipe.unet,VAE=pipe.vae);before={k:digest(v) for k,v in frozen.items()}
  rows=s['engineering_T_array_rows'];raw=torch.load(s['EEG']['path'],weights_only=True,map_location='cpu',mmap=True);assert raw.shape==(16540,63,250);eeg=raw[rows][:,46:63,:].contiguous().cuda();assert eeg.shape==(8,17,250)
  target=torch.load(s['F_targets']['path'],weights_only=True,map_location='cpu',mmap=True)[rows].cuda();latent=np.load(s['latent_bank']['path'],mmap_mode='r');real=torch.from_numpy(np.array(latent[s['engineering_latent_indices']],copy=True)).cuda();assert real.shape==(8,4,64,64)
  gamma=parent.block.layers[0].norm.weight.detach();beta=parent.block.layers[0].norm.bias.detach()
  def conditioned(model,x):
   c=model(x);return ordinary(c,gamma[None].expand_as(c),beta[None].expand_as(c))[0]
  phase='REAL_T_NATIVE_PREFIX';prefixes=[];endpoints=[];native_epsilon=[];tick=time.monotonic()
  for j,row in enumerate(rows):
   states={};hook=pipe.unet.register_forward_hook(lambda module,args,result:states.update(epsilon=result[0].detach().clone()))
   def callback(p,i,t,values):
    if i==2:states['prefix']=values['latents'].detach().clone()
    return values
   seed=int(hashlib.sha256(('HIVE_terminal_profile_T_'+str(row)).encode()).hexdigest()[:8],16)
   with torch.no_grad():
    tokens=P(A(conditioned(parent,eeg[j:j+1]).half()));end=pipe(height=512,width=512,generator=torch.Generator(device='cuda').manual_seed(seed),prompt_embeds=prompts[0],pooled_prompt_embeds=prompts[1],cross_attention_kwargs={'ip_hidden_states':tokens},num_inference_steps=4,guidance_scale=0.,output_type='latent',callback_on_step_end=callback,callback_on_step_end_tensor_inputs=['latents']).images
   hook.remove();prefixes.append(states['prefix']);endpoints.append(end);native_epsilon.append(states['epsilon']);budget()
  torch.cuda.synchronize();prefix_seconds=time.monotonic()-tick;prefix=torch.cat(prefixes).detach().clone();endpoint=torch.cat(endpoints).detach().clone();del states,endpoints,prefixes
  def terminal(z,tokens,grad):
   sch=EulerAncestralDiscreteScheduler.from_config(cfg);sch.set_timesteps(4,device='cuda');t=sch.timesteps[3];scaled=sch.scale_model_input(z,t)
   if grad:eps=checkpoint(lambda x,tt,tk:denoise(pipe,prompts,x,tt,tk),scaled,t,tokens,use_reentrant=False)
   else:eps=denoise(pipe,prompts,scaled,t,tokens)
   return sch.step(eps,t,z,generator=torch.Generator(device='cuda').manual_seed(1729)).prev_sample
  phase='ZERO_UPDATE_GRADIENT_PATH_PARITY';model=scope['BrainMLP'](4250,1024,num_layers=1,dropout=.1).cuda().eval();model.load_state_dict(parent_state['model'],strict=True)
  for name,p in model.named_parameters():p.requires_grad_(not name.startswith('block.layers.0.norm.'))
  assert sum(p.numel() for p in model.parameters() if p.requires_grad)==6450176
  tokens=P(A(conditioned(model,eeg[:1]).half()));got=terminal(prefix[:1],tokens,True);assert torch.equal(got.detach(),endpoint[:1]),'zero-update gradient-enabled final endpoint must equal native single-query endpoint'
  parity_loss=(got.float()-real[:1]).square().mean();grad=torch.autograd.grad(parity_loss,model.proj_in.weight)[0];assert torch.isfinite(grad).all() and float(grad.norm())>0;parity_gradient_norm=float(grad.norm());del got,tokens,grad,model;gc.collect();torch.cuda.empty_cache()
  with torch.no_grad():teacher_tokens=P(A(target.half())).detach().clone()
  phase='FOUR_TEMPORARY_MATCHED_UPDATES';events=[];arm_initial=[]
  for arm in ('real_generated_prefix','forward_Gaussian_same_sigma'):
   model=scope['BrainMLP'](4250,1024,num_layers=1,dropout=.1).cuda().eval();model.load_state_dict(parent_state['model'],strict=True)
   for name,p in model.named_parameters():p.requires_grad_(not name.startswith('block.layers.0.norm.'))
   arm_initial.append(digest(model));opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01);sigma=float(pipe.scheduler.sigmas[3]);rng=torch.Generator(device='cuda').manual_seed(1730);z=prefix if arm=='real_generated_prefix' else (real+sigma*torch.randn(real.shape,generator=rng,device='cuda')).half()
   for step in range(2):
    budget();opt.zero_grad(set_to_none=True);tick=time.monotonic();corrected=conditioned(model,eeg);tok=P(A(corrected.half()));pred=terminal(z,tok,True);endpoint_loss=(pred.float()-real).square().mean();token_loss=(1-torch.nn.functional.cosine_similarity(tok.float(),teacher_tokens.float(),dim=-1)).mean();loss=endpoint_loss+.1*token_loss
    (loss*1024.).backward();gradient={}
    for name,p in model.named_parameters():
     if p.requires_grad:assert p.grad is not None;p.grad.div_(1024.);assert torch.isfinite(p.grad).all();gradient[name]=float(p.grad.norm())
    norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.));assert gradient['proj_in.weight']>0 and gradient['block.layers.0.ff.w1.weight']>0 and gradient['block.layers.0.ff.w2.weight']>0
    opt.step();torch.cuda.synchronize();elapsed=time.monotonic()-tick;events.append(dict(arm=arm,temporary_step=step+1,batch=8,endpoint_MSE=float(endpoint_loss),token_loss=float(token_loss),loss=float(loss),unclipped_gradient_norm=norm,parameter_gradient_norms=gradient,seconds=elapsed,peak_GPU_allocated=torch.cuda.max_memory_allocated()));write(root/f'{arm}_step{step+1}.json',events[-1]);print(json.dumps(events[-1]),flush=True)
   assert torch.equal(model.block.layers[0].norm.weight,gamma) and torch.equal(model.block.layers[0].norm.bias,beta);del opt,model,corrected,tok,pred,loss,endpoint_loss,token_loss;gc.collect();torch.cuda.empty_cache()
  assert arm_initial[0]==arm_initial[1]==before['parent'];assert before=={k:digest(v) for k,v in frozen.items()};checked(s['ordinary']);budget()
  result=dict(status='PASS_NATIVE_TERMINAL_EEG_T8_ENGINEERING_ONLY',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol=ref(path),loader=loader,host=host,all240_GPU_terminal_transitions_bit_exact=True,retained_CPU_backend_precision_exception=True,native_T_queries=8,native_UNet_calls=32,zero_update_native_endpoint_bit_exact=True,zero_update_EEG_input_weight_gradient_norm=parity_gradient_norm,trainable_existing_EEG_parameters=6450176,output_LN_affine_frozen=True,ordinary_frozen=True,all_frozen_weights_unchanged=True,temporary_updates=4,temporary_updates_discarded=True,temporary_weights_saved=False,events=events,prefix_generation_seconds=prefix_seconds,peak_GPU_allocated=torch.cuda.max_memory_allocated(),extra_CPU_peak=memory()-base,seconds=time.monotonic()-started,formal_fit_admitted=False,research_images=0,normal_inference_query_target_access=False,full_chain_OOF=False,NEW_contacted=False)
  write(root/'completion.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('loader','events')}))
 except BaseException as error:write(root/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-started));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);run(p.parse_args().protocol)
