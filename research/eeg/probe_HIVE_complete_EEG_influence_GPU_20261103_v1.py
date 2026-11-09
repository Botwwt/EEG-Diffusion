"""Frozen full-UNet EEG swaps on fixed noisy states and two coupled trajectories."""
import argparse,gc,hashlib,inspect,os,sys,time,traceback
from pathlib import Path
import numpy as np

LAYER_KEYS=('Q_correct_change_energy','Q_donor_change_energy','base_pair_energy','field_pair_energy','pair_dot',
            'pair_change_energy','pair_pre_change_energy','FP16_pair_bias_rounding_energy','Q_correct_energy','Q_donor_energy')

def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--job-name',required=True);a=p.parse_args();started=time.monotonic()
 import json
 spec=json.loads(Path(a.protocol).read_bytes());sys.path.insert(0,str(Path(spec['base_protocol']['path']).parent))
 import brainhive_functional_mean_contract_20261103_v1 as G
 base=G.setup(G.checked(spec['base_protocol']))
 for pin in spec['source_pins']:G.checked(pin)
 G.wrapper(base,a.job_name)
 root=Path(spec['output_root']);code=Path(a.protocol).parent
 def resources(start=False):
  old=G.resources(base,start=start);used=sum(p.stat().st_blocks*512 for directory in (root,code) for p in directory.rglob('*') if p.is_file())
  assert used<=256<<20
  assert old['external']['free_bytes']>=old['required_with_all_prior_commitments']+max(0,(256<<20)-used)
  return dict(existing=old,new_used=used,new_cap=256<<20)
 before=resources(True);rss=G.rss();out=root/'probe';assert not out.exists();out.mkdir(parents=True);stage='LOAD'
 import torch
 from brainhive_component_generator_v1 import load_native,denoise
 from brainhive_bounded_component_v1 import BoundedConditionAdapter
 from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
 def budget(refresh=False):
  assert time.monotonic()-started<870 and G.rss()-rss<10<<30 and torch.cuda.max_memory_allocated()<20<<30
  if refresh:resources()
 def digest(m):
  h=hashlib.sha256()
  for key,value in m.state_dict().items():h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
  return h.hexdigest()
 class Observer:
  def __init__(self,pipe,field):
   self.nodes={name:pipe.unet.get_submodule(name[:-10]) for name,proc in pipe.unet.attn_processors.items() if hasattr(proc,'to_k_ip')}
   assert len(self.nodes)==70
   self.names=list(self.nodes);self.handles=[];self.active=False;self.collect=False;self.records={};self.order=[]
   self.bias={name:torch.from_numpy(field['ordinary__'+name].copy()).cuda().half() for name in self.names}
   for name,m in self.nodes.items():
    self.handles.append(m.to_q.register_forward_hook(self.qhook(name)))
    self.handles.append(m.register_forward_hook(self.ohook(name)))
  def reset(self,active,collect=True):self.active=active;self.collect=collect;self.records={};self.order=[]
  def qhook(self,name):
   def hook(m,args,value):
    if self.collect:
     ids=[0,value.shape[1]//2,value.shape[1]-1];self.records[name]={'Q':value[:,ids].detach().clone()}
   return hook
  def ohook(self,name):
   def hook(m,args,value):
    self.order.append(name);assert value.ndim in (3,4)
    b=self.bias[name][None,None,:] if value.ndim==3 else self.bias[name][None,:,None,None]
    changed=value+b if self.active else value
    if self.collect:
     pre=value if value.ndim==3 else value.flatten(2).transpose(1,2)
     post=changed if changed.ndim==3 else changed.flatten(2).transpose(1,2)
     ids=[0,pre.shape[1]//2,pre.shape[1]-1]
     self.records[name].update(pre=pre[:,ids].detach().clone(),post=post[:,ids].detach().clone())
    return changed if self.active else None
   return hook
  def proof(self):assert len(self.order)==len(set(self.order))==70 and set(self.order)==set(self.names)
  def close(self):
   for h in self.handles:h.remove()
 observer=None
 try:
  torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  torch.cuda.set_per_process_memory_fraction((20<<30)/torch.cuda.get_device_properties(0).total_memory);torch.cuda.reset_peak_memory_stats()
  prepared=G.read(root/'inputs/completion.json');data=dict(np.load(G.checked(prepared['arrays']),allow_pickle=False));rows=spec['queries'];assert len(rows)==80
  saved=torch.load(G.checked(base['ordinary']),weights_only=True,map_location='cpu');head=BoundedConditionAdapter().eval().requires_grad_(False);head.load_state_dict(saved['model'],strict=True);head=head.cuda()
  with torch.inference_mode():
   c=torch.from_numpy(data['condition']).cuda();w=torch.from_numpy(data['weight']).cuda();b=torch.from_numpy(data['bias']).cuda()
   changed=head(c.reshape(160,1024),w.reshape(160,1024),b.reshape(160,1024))[0].reshape(80,2,1024)
   assert torch.allclose(changed.cpu(),torch.from_numpy(data['corrected_CPU']),atol=1e-5,rtol=5e-6)
  loader_dir=out/'loader';loader_dir.mkdir();loader=ConvertedWeightAudit(torch,loader_dir);loader.install()
  try:pipe,A,P,prompts,host=load_native(training=False)
  finally:loader.restore()
  load_proof=loader.complete();frozen={k:digest(m) for k,m in dict(UNet=pipe.unet,VAE=pipe.vae,A=A,P=P).items()}
  fields=dict(np.load(G.checked(spec['bias_fields']),allow_pickle=False))
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
   tokens=P(A(changed.reshape(160,1024).half())).reshape(80,2,4,2048)
   teachers=P(A(torch.from_numpy(data['target']).cuda().reshape(160,1024).half())).reshape(80,2,4,2048)
  np.save(out/'tokens.npy',torch.cat((tokens,teachers),dim=1).cpu().numpy(),allow_pickle=False)
  observer=Observer(pipe,fields);names=observer.names;selected=[0,14,28,42,56,69]
  widths=[len(observer.bias[name]) for name in names]
  pred=np.lib.format.open_memmap(out/'EEG_predictions.npy',mode='w+',dtype=np.float16,shape=(3,80,4,4,4,64,64))
  teacher_pred=np.lib.format.open_memmap(out/'DDPM_teacher_predictions.npy',mode='w+',dtype=np.float16,shape=(80,4,2,4,64,64))
  states=np.lib.format.open_memmap(out/'states.npy',mode='w+',dtype=np.float16,shape=(3,80,4,4,64,64))
  noise_store=np.lib.format.open_memmap(out/'DDPM_noise.npy',mode='w+',dtype=np.float32,shape=(80,4,4,64,64))
  reference=np.lib.format.open_memmap(out/'rollout_predictions.npy',mode='w+',dtype=np.float16,shape=(2,80,4,4,64,64))
  finals=np.empty((2,80,4,64,64),np.float16)
  layer=np.empty((3,80,4,70,len(LAYER_KEYS)),np.float64)
  snapshots=np.zeros((2,3,4,4,6,3,1280,3),np.float16)
  parity=None;noise_proofs=[];first_name=None;actual_order=None;forward_samples=0
  def tensor_sha(value):return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
  def rollout(tok,seed_list,active):
   captured=[];pending={};noises=[];initial=[];scheduler_module=inspect.getmodule(type(pipe.scheduler));original_rng=scheduler_module.randn_tensor
   original_prepare=pipe.prepare_latents;original_step=pipe.scheduler.step
   def capture_rng(*args,**kwargs):
    value=original_rng(*args,**kwargs);noises.append(tensor_sha(value));return value
   def prepare(*args,**kwargs):
    value=original_prepare(*args,**kwargs);initial.append(tensor_sha(value));return value
   def pre(m,args,kwargs):
    observer.reset(active,False);pending['sample']=(args[0] if args else kwargs['sample']).detach().clone();pending['t']=(args[1] if len(args)>1 else kwargs['timestep']).detach().clone()
   def post(m,args,kwargs,output):
    observer.proof();value=output[0] if isinstance(output,tuple) else output.sample
    captured.append(dict(**pending,prediction=value.detach().clone()));pending.clear()
   hs=[pipe.unet.register_forward_pre_hook(pre,with_kwargs=True),pipe.unet.register_forward_hook(post,with_kwargs=True)]
   scheduler_module.randn_tensor=capture_rng;pipe.prepare_latents=prepare
   try:
    generators=[torch.Generator(device='cuda').manual_seed(seed) for seed in seed_list]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
     value=pipe(height=512,width=512,generator=generators,prompt_embeds=prompts[0].expand(len(tok),-1,-1),
       pooled_prompt_embeds=prompts[1].expand(len(tok),-1),cross_attention_kwargs={'ip_hidden_states':tok},num_inference_steps=4,guidance_scale=0.,output_type='latent').images
    assert len(captured)==4 and len(noises)==4 and len(initial)==1 and pipe.scheduler.step==original_step
    return value,captured,dict(initial=initial[0],ancestral=noises,step_signature_unchanged=True)
   finally:
    scheduler_module.randn_tensor=original_rng;pipe.prepare_latents=original_prepare
    for h in hs:h.remove()
  def evaluate(sample,timestep,tok,active):
   nonlocal first_name,actual_order,forward_samples
   observer.reset(active,True)
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):result=denoise(pipe,prompts,sample,timestep,tok)
   observer.proof();forward_samples+=len(tok)
   if actual_order is None:actual_order=list(observer.order);first_name=actual_order[0]
   assert observer.order==actual_order
   return result.detach().cpu().numpy(),observer.records
  from diffusers import DDPMScheduler
  ddpm=DDPMScheduler.from_config(pipe.scheduler.config);pipe.scheduler.set_timesteps(4,device='cuda');times=pipe.scheduler.timesteps.clone()
  stage='BOTH_REAL_TRAJECTORIES_AND_FULL_CONDITION_SWAPS'
  for lo in range(0,80,2):
   budget(refresh=True);indices=list(range(lo,lo+2));seeds=[r['trajectory_seed'] for r in rows[lo:lo+2]]
   if lo==0:
    observer.close();observer=None
    plain,_,plain_proof=rollout_without_observer(pipe,prompts,tokens[:2,0],seeds,torch)
    observer=Observer(pipe,fields)
   ordinary,ordinary_states,p0=rollout(tokens[lo:lo+2,0],seeds,False)
   biased,biased_states,p1=rollout(tokens[lo:lo+2,0],seeds,True)
   if lo==0:
    assert torch.equal(plain,ordinary) and plain_proof==p0
    parity=dict(status='EXACT_FINAL_LATENT_AND_ALL_NOISE_OBSERVER_PARITY',batch=2,latent_sha=tensor_sha(ordinary))
   assert p0==p1;noise_proofs.append(dict(queries=indices,ordinary=p0,biased=p1))
   finals[0,lo:lo+2]=ordinary.cpu().numpy();finals[1,lo:lo+2]=biased.cpu().numpy()
   for ti,timestep in enumerate(times):
    generator=torch.Generator(device='cuda').manual_seed(int(hashlib.sha256(f'1729|fullEEG_DDPM|{lo}|{ti}'.encode()).hexdigest()[:15],16))
    noise=torch.randn((2,4,64,64),generator=generator,device='cuda',dtype=torch.float32)
    noisy=ddpm.add_noise(torch.from_numpy(data['latent'][lo:lo+2]).cuda(),noise,timestep.long().expand(2)).half()
    noise_store[lo:lo+2,ti]=noise.cpu().numpy()
    reference[0,lo:lo+2,ti]=ordinary_states[ti]['prediction'].cpu().numpy();reference[1,lo:lo+2,ti]=biased_states[ti]['prediction'].cpu().numpy()
    for domain,sample in enumerate((noisy,ordinary_states[ti]['sample'],biased_states[ti]['sample'])):
     budget();states[domain,lo:lo+2,ti]=sample.cpu().numpy();captures=[]
     for case,(which,active) in enumerate(((0,False),(1,False),(0,True),(1,True))):
      value,record=evaluate(sample,timestep,tokens[lo:lo+2,which],active);pred[domain,lo:lo+2,ti,case]=value;captures.append(record)
     if domain==1:assert np.array_equal(pred[domain,lo:lo+2,ti,0],reference[0,lo:lo+2,ti])
     if domain==2:assert np.array_equal(pred[domain,lo:lo+2,ti,2],reference[1,lo:lo+2,ti])
     for name in names:
      ci=[{key:value.detach().cpu().double().numpy() for key,value in rec[name].items()} for rec in captures]
      q0,qd,qb,qbd=[v['Q'] for v in ci];a,d,ac,dc=[v['post'] for v in ci];ap,dp=[ci[j]['pre'] for j in (2,3)]
      old=a-d;new=ac-dc;pre=ap-dp
      if name==first_name:
       assert np.array_equal(q0,qb) and np.array_equal(qd,qbd)
       assert np.array_equal(ci[0]['pre'],ci[2]['pre']) and np.array_equal(ci[1]['pre'],ci[3]['pre'])
      def energy(x):return np.mean(x*x,axis=(1,2))
      layer[domain,lo:lo+2,ti,names.index(name)]=np.stack((energy(qb-q0),energy(qbd-qd),energy(old),energy(new),np.mean(old*new,axis=(1,2)),
        energy(new-old),energy(pre-old),energy(new-pre),energy(q0),energy(qd)),axis=1)
      if name in [names[k] for k in selected]:
       j=selected.index(names.index(name));width=widths[names.index(name)]
       for offset,global_index in enumerate(indices):
        if global_index not in (0,4):continue
        si=(0,4).index(global_index)
        for case,rec in enumerate(ci):
         for vi,key in enumerate(('Q','pre','post')):snapshots[si,domain,ti,case,j,:,:width,vi]=rec[key][offset].astype(np.float16)
     if domain==0:
      for j in range(2):teacher_pred[lo:lo+2,ti,j]=evaluate(sample,timestep,teachers[lo:lo+2,j],False)[0]
   del ordinary_states,biased_states
   print(json.dumps(dict(stage=stage,completed=lo+2,expected=80,seconds=time.monotonic()-started)),flush=True)
  assert forward_samples==4480
  for value in (pred,teacher_pred,states,noise_store,reference):value.flush()
  np.save(out/'final_latents.npy',finals,allow_pickle=False);np.save(out/'layer_statistics.npy',layer,allow_pickle=False);np.save(out/'fixed_raw_layer_samples.npy',snapshots,allow_pickle=False)
  G.write(out/'trace_metadata.json',dict(rows=rows,timesteps=times.cpu().tolist(),nodes=names,actual_forward_order=actual_order,widths=widths,layer_keys=LAYER_KEYS,
    raw_query_indices=[0,4],raw_node_indices=selected,raw_views=['Q','pre','post'],raw_positions='0,middle,last',noise_proofs=noise_proofs,parity=parity))
  observer.close();observer=None
  assert {k:digest(m) for k,m in dict(UNet=pipe.unet,VAE=pipe.vae,A=A,P=P).items()}==frozen
  budget(True)
  outputs={p.name:G.ref(p) for p in out.glob('*.npy')}
  assert set(outputs)=={'tokens.npy','EEG_predictions.npy','DDPM_teacher_predictions.npy','states.npy','DDPM_noise.npy','rollout_predictions.npy','final_latents.npy','layer_statistics.npy','fixed_raw_layer_samples.npy'}
  complete=dict(status='COMPLETE_FROZEN_FULL_UNET_ALL10_SUBJECT80_T_THREE_DOMAINS',UTC=G.datetime.datetime.now(G.datetime.timezone.utc).isoformat(),
    protocol=G.ref(a.protocol),source=G.ref(__file__),outputs=outputs,trace=G.ref(out/'trace_metadata.json'),inputs=G.ref(root/'inputs/completion.json'),loader=load_proof,
    host=host,all_original_weights_unchanged=True,all5120_sample_UNet_forwards=True,evaluation_sample_forwards=4480,trajectory_sample_forwards=640,
    additional_observer_parity_sample_forwards=8,upstream_true_OOF=True,whole_chain_OOF=False,parameters_updated=0,generated_images=0,
    seconds=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=G.rss()-rss,resources_before=before,resources_after=resources(),NEW_contacted=False)
  G.write(out/'completion.json',complete);print(json.dumps(complete),flush=True)
 except BaseException as error:
  G.write(out/'failure.json',dict(stage=stage,error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-started,automatic_retry=False));raise
 finally:
  if observer is not None:observer.close()

def rollout_without_observer(pipe,prompts,tokens,seeds,torch):
 module=inspect.getmodule(type(pipe.scheduler));rng=module.randn_tensor;prepare=pipe.prepare_latents;noise=[];initial=[];states=[]
 def random(*args,**kwargs):
  v=rng(*args,**kwargs);noise.append(hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest());return v
 def start(*args,**kwargs):
  v=prepare(*args,**kwargs);initial.append(hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest());return v
 module.randn_tensor=random;pipe.prepare_latents=start
 try:
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
   result=pipe(height=512,width=512,generator=[torch.Generator(device='cuda').manual_seed(s) for s in seeds],prompt_embeds=prompts[0].expand(len(tokens),-1,-1),
    pooled_prompt_embeds=prompts[1].expand(len(tokens),-1),cross_attention_kwargs={'ip_hidden_states':tokens},num_inference_steps=4,guidance_scale=0.,output_type='latent').images
  assert len(initial)==1 and len(noise)==4
  return result,states,dict(initial=initial[0],ancestral=noise,step_signature_unchanged=True)
 finally:module.randn_tensor=rng;pipe.prepare_latents=prepare
if __name__=='__main__':main()
