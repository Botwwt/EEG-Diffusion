"""Fixed T80, frozen native four-step rendering and repeated-EEG target geometry.

Engineering/target construction only: no fitted EEG model, optimizer update,
new adapter, saved research picture, held-out score or checkpoint selection.
"""
import argparse,datetime,gc,hashlib,importlib.util,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
def read(p):return json.loads(Path(p).read_bytes())
def ref(p):
 p=Path(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
def checked(x):
 got=ref(x['path']);assert got['sha256']==x['sha256']
 if 'bytes' in x:assert got['bytes']==x['bytes']
 return Path(x['path'])
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def memory():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
def main(path):
 s=read(path);assert s['host']=='old' and s['status']=='REGISTERED_T80_EQUAL_ANGLE_EEG_TARGET_DIAGNOSTIC'
 assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==s['GPU_UUID']
 assert 'run_gpu_job_v2.py' in (Path('/proc')/str(os.getppid())/'cmdline').read_text()
 for x in s['source_pins']+s['payloads']:checked(x)
 out=Path(s['output_root']);out.mkdir();started=time.monotonic();base=memory();phase='IMPORTS'
 sys.path.insert(0,s['repository_root']);import brainhive_paired_study_contract_20261025_v1 as C
 C.setup_runtime(dict(runtime_manifest=s['runtime_manifest']))
 import torch,numpy as np
 import torch.nn.functional as F
 from torch.utils.checkpoint import checkpoint
 from diffusers import EulerAncestralDiscreteScheduler
 import brainhive_native_models_v1 as N
 from brainhive_bounded_component_v1 import BoundedConditionAdapter
 from brainhive_component_generator_v1 import load_native,denoise
 from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
 from brainhive_renderable_EEG_target_20261029_v1 import empirical_repeat_precision,spherical_quadratic_target,centered_basis
 torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.cuda.set_per_process_memory_fraction(min(1.,float(20<<30)/torch.cuda.get_device_properties(0).total_memory))
 def budget():assert time.monotonic()-started<800 and memory()-base<10<<30 and torch.cuda.max_memory_allocated()<20<<30 and shutil.disk_usage(out).free>3<<30
 def digest(module):
  h=hashlib.sha256()
  for n,v in module.state_dict().items():h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
  return h.hexdigest()
 try:
  phase='EXISTING_FROZEN_CLIP_VISUAL';spec=importlib.util.spec_from_file_location('pinned_clip_model',s['CLIP_source']['path']);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
  jit=torch.jit.load(s['CLIP_weights']['path'],map_location='cpu').eval();state={k[len('visual.'):]:v for k,v in jit.state_dict().items() if k.startswith('visual.')}
  critic=mod.VisionTransformer(224,14,1024,24,16,768).eval().requires_grad_(False);critic.load_state_dict(state,strict=True);critic=critic.cuda().float();del state,jit;gc.collect();budget()
  phase='EXACT_NATIVE_HOST';(out/'loader').mkdir();audit=ConvertedWeightAudit(torch,out/'loader');audit.install()
  try:pipe,A,P,prompts,host=load_native(training=False)
  finally:audit.restore()
  loading=audit.complete();ordinary=BoundedConditionAdapter(predict_risk=False).cuda().eval().requires_grad_(False);ordinary.load_state_dict(torch.load(s['ordinary']['path'],weights_only=True,map_location='cpu')['model'],strict=True)
  parent=N.classes()['BrainMLP'](4250,1024,num_layers=1,dropout=.1).cuda().eval().requires_grad_(False);parent.load_state_dict(torch.load(s['parent']['path'],weights_only=True,map_location='cpu')['model'],strict=True)
  frozen=dict(parent=parent,ordinary=ordinary,A=A,P=P,UNet=pipe.unet,VAE=pipe.vae,critic=critic);before={k:digest(v) for k,v in frozen.items()}
  gamma=parent.block.layers[0].norm.weight.detach();beta=parent.block.layers[0].norm.bias.detach();cfg=read(s['scheduler_config']['path']);trials=np.load(s['trials']['path'],mmap_mode='r');index=[json.loads(x) for x in Path(s['index']['path']).read_text().splitlines()];T=s['T_array_rows'];assert len(index)==13230 and trials.shape==(13230,4,17,250)
  Fbank=torch.load(s['F_targets']['path'],weights_only=True,map_location='cpu',mmap=True);raw=torch.load(s['EEG']['path'],weights_only=True,map_location='cpu',mmap=True)
  # FP16->FP32 is exact for these frozen values. No irreversible BF16 conversion.
  pipe.vae.to(dtype=torch.float32);pipe.vae.enable_slicing();vae_float_digest=digest(pipe.vae)
  def conditioned(e):
   c=parent(e);return ordinary(c,gamma[None].expand_as(c),beta[None].expand_as(c))[0]
  def render(c,seed,grad=False,features=True):
   scheduler=EulerAncestralDiscreteScheduler.from_config(cfg);scheduler.set_timesteps(4,device='cuda');pipe.scheduler=scheduler
   gen=torch.Generator(device='cuda').manual_seed(seed);z=pipe.prepare_latents(1,4,512,512,prompts[0].dtype,torch.device('cuda'),gen,None);tokens=P(A(c.half()))
   for t in scheduler.timesteps:
    scaled=scheduler.scale_model_input(z,t)
    eps=checkpoint(lambda x,tk,tt:denoise(pipe,prompts,x,tt,tk),scaled,tokens,t,use_reentrant=False) if grad else denoise(pipe,prompts,scaled,t,tokens)
    z=scheduler.step(eps,t,z,generator=gen).prev_sample
   if not features:return z
   def decode(x):return pipe.vae.decode(x.float()/pipe.vae.config.scaling_factor,return_dict=False)[0]
   image=checkpoint(decode,z,use_reentrant=False) if grad else decode(z)
   image=(image*.5+.5).clamp(0,1);image=F.interpolate(image,size=(224,224),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
   mean=image.new_tensor([.48145466,.4578275,.40821073])[None,:,None,None];std=image.new_tensor([.26862954,.26130258,.27577711])[None,:,None,None];image=(image-mean)/std
   value=checkpoint(critic,image,use_reentrant=False) if grad else critic(image)
   return F.normalize(value.float(),dim=-1),z
  def c_from_u(u):return beta[None]+gamma[None]*u
  records=[];bank={};Q=centered_basis(1024);phase='FIXED_T80_NATIVE_GEOMETRY'
  def equal_angle(main,other,origin):
   radius=np.linalg.norm(origin);cosine=float(np.clip(origin@main/radius**2,-1.,1.));direction=other-origin*(origin@other/radius**2);length=np.linalg.norm(direction)
   if length<1e-12:return main.copy(),True
   value=cosine*origin+np.sqrt(max(0.,1-cosine*cosine))*radius*direction/length
   assert abs(value.mean())<1e-8 and abs(np.linalg.norm(value)-radius)/radius<1e-7
   assert abs((value-origin)@(value-origin)-(main-origin)@(main-origin))<1e-5
   return value,False
  for j,pos in enumerate(s['T_positions']):
   budget();tick=time.monotonic();row=T[pos];assert index[pos]['array_row']==row
   x=torch.from_numpy(np.array(trials[pos],copy=True)).cuda();mean=x.mean(0,keepdim=True);expected=raw[row:row+1,46:63,:].cuda();assert torch.max((mean-expected).abs())<3e-5
   views=(x.sum(0,keepdim=True)-x)/3.
   with torch.no_grad():c=conditioned(mean);loo=conditioned(views);m=((c-beta)/gamma)[0].double().cpu().numpy();v=((loo-beta)/gamma).double().cpu().numpy();f=((Fbank[row].cuda()-beta)/gamma).double().cpu().numpy()
   m-=m.mean();f-=f.mean();f*=np.linalg.norm(m)/np.linalg.norm(f);precision,proxy=empirical_repeat_precision(v)
   D=np.column_stack([m-f,(v-v.mean(0)).T]);D-=D.mean(0);B,sv,_=np.linalg.svd(D,full_matrices=False);k=int((sv>sv[0]*1e-8).sum());B=B[:,:k];assert 1<=k<=4
   seed=int(hashlib.sha256(('HIVE_renderable_target_T_'+str(row)).encode()).hexdigest()[:8],16)
   with torch.no_grad():
    tokens=P(A(c.half()));native=pipe(height=512,width=512,generator=torch.Generator(device='cuda').manual_seed(seed),prompt_embeds=prompts[0],pooled_prompt_embeds=prompts[1],cross_attention_kwargs={'ip_hidden_states':tokens},num_inference_steps=4,guidance_scale=0.,output_type='latent').images
    manual=render(c,seed,features=False);assert torch.equal(manual,native),'All four manual native steps must be bit exact at zero update'
    cf=c_from_u(torch.from_numpy(f[None]).cuda().float());visual,zF=render(cf,seed);EEG,zM=render(c,seed)
    J=[];h=.125
    for a in range(k):
     d=torch.from_numpy(B[:,a][None]).cuda().float();plus,_=render(c_from_u(torch.from_numpy(f[None]).cuda().float()+h*d),seed);minus,_=render(c_from_u(torch.from_numpy(f[None]).cuda().float()-h*d),seed);J.append(((plus-minus)/(2*h))[0].double().cpu().numpy());budget()
   J=np.column_stack(J);H=B@(J.T@J)@B.T;trace=float(np.trace(H));assert trace>1e-12;H*=k/trace
   target,certificate=spherical_quadratic_target(m,f,H,precision);isotropic,iso_certificate=spherical_quadratic_target(m,f,np.eye(1024)*(k/1023.),precision)
   permutation=np.random.default_rng(1729).permutation(1024);permuted=precision[permutation][:,permutation];wrong,wrong_certificate=spherical_quadratic_target(m,f,H,permuted)
   geodesic,e0=equal_angle(target,f,m);eq_iso,e1=equal_angle(target,isotropic,m);eq_wrong,e2=equal_angle(target,wrong,m)
   with torch.no_grad():
    phi,_=render(c_from_u(torch.from_numpy(target[None]).cuda().float()),seed);phii,_=render(c_from_u(torch.from_numpy(isotropic[None]).cuda().float()),seed);phiw,_=render(c_from_u(torch.from_numpy(wrong[None]).cuda().float()),seed)
    pg,_=render(c_from_u(torch.from_numpy(geodesic[None]).cuda().float()),seed);pi,_=render(c_from_u(torch.from_numpy(eq_iso[None]).cuda().float()),seed);pw,_=render(c_from_u(torch.from_numpy(eq_wrong[None]).cuda().float()),seed)
    raw_visual,_=render(Fbank[row:row+1].cuda(),seed)
   def semantic(p):return float((p-visual).square().sum())
   rec=dict(position=pos,array_row=row,seed=seed,rank=k,native_zero_update_all4steps_bit_exact=True,repeat_dispersion=proxy,
    local_metric_trace_before_normalization=trace,active_curvature_trace_after_normalization=float(np.trace(H)),
    semantic_squared_distance_to_rendered_feasible_visual=dict(EEG=semantic(EEG),coupled=semantic(phi),isotropic=semantic(phii),permuted_repeat_geometry=semantic(phiw),equal_angle_visual_geodesic=semantic(pg),equal_angle_isotropic=semantic(pi),equal_angle_permuted_repeat=semantic(pw),unprojected_visual=semantic(raw_visual)),
    distance_to_EEG=dict(visual=float(np.sum((f-m)**2)),coupled=float(np.sum((target-m)**2)),isotropic=float(np.sum((isotropic-m)**2)),permuted_repeat_geometry=float(np.sum((wrong-m)**2))),
    target_certificate=certificate,isotropic_certificate=iso_certificate,permuted_certificate=wrong_certificate,concept_id=index[pos]['concept_id'],equal_angle_fallback=[e0,e1,e2],equal_angle_squared_distances=[float(np.sum((x-m)**2)) for x in (target,geodesic,eq_iso,eq_wrong)],seconds=time.monotonic()-tick)
   bank.update({f'{j}_m':m,f'{j}_f':f,f'{j}_views':v,f'{j}_basis':B,f'{j}_J':J,f'{j}_target':target,f'{j}_isotropic':isotropic,f'{j}_wrong':wrong})
   bank.update({f'{j}_geodesic':geodesic,f'{j}_eq_iso':eq_iso,f'{j}_eq_wrong':eq_wrong,f'{j}_phi_visual':visual.cpu().numpy(),f'{j}_phi_main':phi.cpu().numpy(),f'{j}_phi_geodesic':pg.cpu().numpy(),f'{j}_phi_eq_iso':pi.cpu().numpy(),f'{j}_phi_eq_wrong':pw.cpu().numpy()})
   # One real input-gradient check, through all four native steps, VAE and CLIP.
   if j==0:
    phase='ALL4_NATIVE_STEPS_SEMANTIC_INPUT_GRADIENT';uc=torch.from_numpy(m[None]).cuda().float().requires_grad_(True);pf,_=render(c_from_u(uc),seed,grad=True);loss=(pf-visual.detach()).square().sum();g=torch.autograd.grad(loss*1024.,uc)[0]/1024.;assert torch.isfinite(g).all() and float(g.norm())>0
    rec['all4step_semantic_input_gradient_norm']=float(g.norm());rec['all4step_semantic_loss']=float(loss);bank['semantic_gradient']=g.detach().cpu().numpy();del uc,pf,loss,g;gc.collect();torch.cuda.empty_cache();phase='FIXED_T80_NATIVE_GEOMETRY'
   torch.cuda.synchronize();rec['peak_GPU_allocated']=torch.cuda.max_memory_allocated();records.append(rec);write(out/f'T{j:02d}.json',rec);print(json.dumps(rec),flush=True);budget()
  assert len(records)==80 and len({x['concept_id'] for x in records})==8
  concepts=sorted({x['concept_id'] for x in records});groups={c:[x for x in records if x['concept_id']==c] for c in concepts};assert all(len(x)==10 for x in groups.values())
  samples=np.random.default_rng(1729).integers(0,8,size=(1000,8));comparisons={}
  for control in ('equal_angle_visual_geodesic','equal_angle_isotropic','equal_angle_permuted_repeat'):
   differences=np.array([np.mean([x['semantic_squared_distance_to_rendered_feasible_visual'][control]-x['semantic_squared_distance_to_rendered_feasible_visual']['coupled'] for x in groups[c]]) for c in concepts]);interval=np.quantile(differences[samples].mean(1),[.025,.975]).tolist();comparisons[control]=dict(semantic_distance_reduction=float(differences.mean()),descriptive_concept_bootstrap95=interval,positive_concepts=int((differences>0).sum()))
  support=all(v['semantic_distance_reduction']>=.01 and v['descriptive_concept_bootstrap95'][0]>0 for v in comparisons.values())
  write(out/'fixed_T80_target_comparisons.json',dict(queries=80,concepts=8,comparisons=comparisons,prespecified_practical_reduction=.01,target_geometry_support=support,action='TARGET_GEOMETRY_SUPPORTED_FOR_SEPARATE_FIT_REVIEW' if support else 'STOP_THIS_LOCAL_TARGET_GEOMETRY',normal_EEG_image_gain_established=False,description='CLIP distance to privileged rendered feasible F in T; equal EEG angle, NOT heldout reconstruction improvement'))
  assert before['VAE']==digest(pipe.vae.half()),'FP16 weight values must roundtrip exactly';pipe.vae.float();assert vae_float_digest==digest(pipe.vae)
  assert all(before[k]==digest(v) for k,v in frozen.items() if k!='VAE');checked(s['ordinary']);budget();np.savez_compressed(out/'all_T80_geometry.npz',**bank)
  result=dict(status='COMPLETE_FIXED_T80_EQUAL_ANGLE_RENDERABLE_EEG_TARGET_DIAGNOSTIC',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol=ref(path),queries=80,records=records,geometry=ref(out/'all_T80_geometry.npz'),all_frozen_weight_values_unchanged=True,temporary_optimizer_updates=0,new_trainable_parameters=0,new_research_images=0,normal_inference_target_access=False,
   differentiable_float_resize_not_uint8_official_metric=True,CLIP_critic_shared_with_official_CLIP_metric_not_independent=True,primary_DINO_not_used_for_training=True,
   real_EEG_or_image_gain_established=False,formal_fit_admitted=False,full_chain_OOF=False,iid_repeat_noise_not_established=True,
   approximation='local low-rank finite-difference generator semantic metric in mean/repeat span; not full Jacobian or true posterior',
   peak_GPU_allocated=torch.cuda.max_memory_allocated(),extra_CPU_peak=memory()-base,seconds=time.monotonic()-started,loader=loading,host=host,NEW_contacted=False)
  write(out/'completion.json',result)
 except BaseException as error:
  write(out/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-started));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);main(p.parse_args().protocol)
