"""Fixed T80 native NN trajectories, common-Q responses of four EEG models."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import fcntl,hashlib,json,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
q=json.loads(Path(sys.argv[1]).read_bytes());root=Path(q['parent_root']);out=Path(q['output_root']);started=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
sys.path.insert(0,str(root/'code'));import atm_fixed_projection_contract_20261105_v190 as C
c=C.setup(root/'code/policy.json');K=c['K'];before=C.resources(c,True);assert K.ref(__file__)==q['sources']['GPU']
job=K.read(c['repo']/'audit'/(q['job_name']+'_job.json'));assert job['status']=='RUNNING' and job['pid']==os.getpid()
lock=(c['repo']/'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==q['GPU_UUID']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
import torch
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction((20<<30)/torch.cuda.get_device_properties(0).total_memory)
from atm_native_training_runtime_20261103_v1 import load_host,tensor_sha
from brainhive_native_direction_contract_20261103_v1 import tensor_digest
from atm_native_response_geometry_20261105_v193_v2 import SameQueryResponse
audit=None
def guard():
 assert time.monotonic()-started<850 and time.time()<q['absolute_deadline_epoch']
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<10<<30 and torch.cuda.max_memory_allocated()<20<<30
 own=K.allocated(root);assert own<384<<20 and K.allocated(out)<q['maximum_output_bytes']
 fs=os.statvfs(root);assert fs.f_bavail*fs.f_frsize>=before['required_including_this_project']+before['own_used_bytes']-own
try:
 d=K.read(K.checked(q['conditions_completion']));assert d['status']=='PASS_ALL320_T80_CPU32_CPU64_FROZEN_CONDITIONS'
 bank=np.load(K.checked(d['conditions']));values=bank['conditions'];teachers=bank['teacher'];ids=bank['T_rows']
 loader=out/'loader';loader.mkdir();pipe,P,prompt,pooled,timeids,diffusion,proof=load_host(c['cv']['bq'],loader)
 import transformers
 assert transformers.__version__=='4.36.0'
 frozen={k:tensor_digest(v,half=True) for k,v in [('UNet',pipe.unet),('VAE',pipe.vae)]}
 import diffusers.schedulers.scheduling_euler_ancestral_discrete as EA
 old_noise,old_prepare=EA.randn_tensor,pipe.prepare_latents;capture={}
 def cn(*args,**kwargs):
  z=old_noise(*args,**kwargs);capture['ancestral'].append(tensor_sha(z));return z
 def ci(*args,**kwargs):
  z=old_prepare(*args,**kwargs);capture['initial'].append(tensor_sha(z));return z
 EA.randn_tensor,pipe.prepare_latents=cn,ci
 def run(value,seed,pil=False):
  global capture
  capture={'initial':[],'ancestral':[]}
  with torch.inference_mode():
   image=pipe(prompt_embeds=prompt,pooled_prompt_embeds=pooled,height=512,width=512,ip_adapter_image_embeds=[value[:,None].half()],num_inference_steps=4,guidance_scale=0.,generator=torch.Generator(device='cuda').manual_seed(seed),output_type='pil' if pil else 'latent').images
  assert len(capture['initial'])==1 and len(capture['ancestral'])==4
  return image[0],dict(capture)
 # Pre-existing complete Dcheck q0 is an engineering regression, not a new score.
 ref=c['positive'][0];oldim,oldn=run(torch.tensor(c['values'][0,0][None],device='cuda'),ref['image_seed'],True)
 assert hashlib.sha256(np.asarray(oldim.convert('RGB')).tobytes()).hexdigest()==ref['pixels_sha256'] and oldn==ref['noise']
 oldim.save(out/'historical_NN_pixel_replay.webp',format='WEBP',lossless=True,quality=100)
 # Unmodified T0 image establishes the same-worker hook identity below.
 t0,t0noise=run(torch.tensor(values[0,0][None],device='cuda'),1729+int(ids[0]),True)
 t0hash=hashlib.sha256(np.asarray(t0.convert('RGB')).tobytes()).hexdigest()
 tokens=np.empty((80,4,4,2048),np.float16);vtokens=np.empty((80,4,2048),np.float16)
 with torch.inference_mode():
  for i in range(80):
   for j in range(4):tokens[i,j]=P(torch.tensor(values[i,j][None],device='cuda').half()).cpu().numpy()[0]
   vtokens[i]=P(torch.tensor(teachers[i][None],device='cuda').half()).cpu().numpy()[0]
 savedtokens=np.load(K.checked(q['tokens_reuse']));assert np.array_equal(savedtokens['EEG'],tokens) and np.array_equal(savedtokens['visual'],vtokens)
 distance=np.zeros((80,4,70,4,10,3),np.float32);scale=np.zeros((80,4,70,3),np.float32)
 energy=np.zeros((80,4,70,3,3),np.float32);alignment=np.zeros_like(energy);coverage=np.zeros((80,4,70),np.uint8);records=[];maxerr=maxrisk=0.
 audit=SameQueryResponse(pipe)
 for i in range(80):
  guard();group=(i//10)*10;tt=torch.tensor(np.concatenate([tokens[i],vtokens[group:group+10]]),device='cuda')
  audit.begin(i,tt);im,noise=run(torch.tensor(values[i,0][None],device='cuda'),1729+int(ids[i]),i==0)
  rows=audit.end()
  if i==0:
   px=hashlib.sha256(np.asarray(im.convert('RGB')).tobytes()).hexdigest();assert px==t0hash and noise==t0noise
   im.save(out/'T0_hook_identity.webp',format='WEBP',lossless=True,quality=100);terminal=dict(pixels_sha256=px)
  else:terminal=dict(final_latent_FP16_sha256=tensor_sha(im))
  for t,l,d,s,e,al,err,risk in rows:
   coverage[i,t,l]+=1;distance[i,t,l]=d;scale[i,t,l]=s;energy[i,t,l]=e;alignment[i,t,l]=al;maxerr=max(maxerr,err);maxrisk=max(maxrisk,risk)
  records.append(dict(query=i,array_row=int(ids[i]),teacher_rows=ids[group:group+10].tolist(),correct_teacher_index=i%10,seed=1729+int(ids[i]),noise=noise,**terminal))
  K.write(out/'progress.json',dict(complete_queries=i+1,target_queries=80,actual_attention_units=int(coverage.sum()),new_updates=0))
 assert np.all(coverage==1) and len(audit.cases)==8
 np.savez_compressed(out/'responses.npz',distance=distance,scale=scale,energy=energy,alignment=alignment,coverage=coverage)
 np.savez_compressed(out/'independent_cases.npz',**{label+'__'+key:val for label,case in audit.cases.items() for key,val in case.items()})
 K.write(out/'trajectories.json',records);K.write(out/'actual_KV_layouts.json',dict(layouts=sorted(audit.layouts),all44800_KV_values_bitexact_after_reshape=True));audit.close();audit=None
 assert frozen=={k:tensor_digest(v,half=True) for k,v in [('UNet',pipe.unet),('VAE',pipe.vae)]}
 guard();K.write(out/'GPU_completion.json',dict(status='COMPLETE_T80_NATIVE_NN_SAME_Q_22400_UNITS_FOUR_MODELS_TEN_TEACHERS',responses=K.ref(out/'responses.npz'),cases=K.ref(out/'independent_cases.npz'),tokens=q['tokens_reuse'],trajectories=K.ref(out/'trajectories.json'),conditions=d['conditions'],
  units=22400,query_positions_per_unit=16,all70_layers_all4_times=True,NN_generated_T_trajectories=80,additional_regression_trajectories=2,UNet_NFE=328,VAE_decoded_images=3,saved_engineering_images=2,new_effect_images=0,updates=0,
  maximum_native_SDPA_relative_error=maxerr,maximum_risk_identity_error=maxrisk,all_host_weights_unchanged=True,historical_pixel_noise_exact=True,T0_hook_pixel_noise_exact=True,loader=proof,
  seconds=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0))
except BaseException as exc:
 K.write(out/'GPU_failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-started));raise
finally:
 if audit is not None:audit.close()
 if 'EA' in globals():EA.randn_tensor,pipe.prepare_latents=old_noise,old_prepare
