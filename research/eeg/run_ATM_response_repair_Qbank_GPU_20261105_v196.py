"""Capture immutable native NN Q on four predetermined T stimuli, all70x4."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import argparse,fcntl,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import atm_response_repair_contract_20261105_v196 as C
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--mode',choices=['Qbank'],required=True);a=ap.parse_args()
c=C.setup(a.policy);q,K=c['q'],c['K'];out=Path(q['project_root'])/'Qbank';started=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
assert K.ref(__file__)==q['sources']['Qbank'];before=C.resources(c,True)
job=K.read(c['repo']/'audit'/(q['job_prefix']+'_Qbank_job.json'));assert job['pid']==os.getpid() and job['status']=='RUNNING'
lock=(c['repo']/'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==c['physical']['GPU_UUID']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
import torch
import torch.nn.functional as F
from safetensors import safe_open
from atm_native_training_runtime_20261103_v1 import load_host,tensor_sha
from brainhive_native_direction_contract_20261103_v1 import tensor_digest
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction((20<<30)/torch.cuda.get_device_properties(0).total_memory)
handles=[];original=F.scaled_dot_product_attention;active=False;pending=None;counts=None;current=None;bank={};proofs=[]
def guard():
 assert time.monotonic()-started<850 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<10<<30
 assert torch.cuda.max_memory_allocated()<20<<30;used=K.allocated(q['project_root']);assert used<q['maximum_output_bytes']
 fs=os.statvfs(out);assert fs.f_bavail*fs.f_frsize>=before['required_including_this_project']+before['own_used_bytes']-used
def hook(i):
 def capture(module,args,output):
  global pending
  if active:assert pending is None;pending=i
 return capture
def sdpa(Q,k,v,*args,**kwargs):
 global pending
 z=original(Q,k,v,*args,**kwargs)
 if pending is None:return z
 assert Q.shape[0]==1 and k.shape[-2]==v.shape[-2]==4
 i=pending;t=counts[i];assert t<4;counts[i]+=1;pending=None
 small=Q[:,:,[0,-1],:];bank[f'l{i:02d}'][current,t]=small[0].cpu().numpy()
 return z
try:
 src=np.load(K.checked(q['T80_conditions']));values=src['conditions'];ids=src['T_rows'];assert values.shape==(80,4,1024)
 (out/'loader').mkdir();pipe,projection,prompt,pooled,_,_,loader=load_host(c['cv']['bq'],out/'loader')
 frozen={n:tensor_digest(v,half=True) for n,v in [('UNet',pipe.unet),('VAE',pipe.vae)]}
 layers=[(n,p) for n,p in pipe.unet.attn_processors.items() if hasattr(p,'to_k_ip')];assert len(layers)==70
 # Verify every source-file K/V tensor against the actual loaded processor.
 h94=c['cv']['bq']['h94'];K.checked(h94);layouts=[]
 with safe_open(h94['path'],framework='pt',device='cpu') as f:
  for i,(name,p) in enumerate(layers):
   assert type(p).__name__=='IPAdapterAttnProcessor2_0' and list(p.num_tokens)==[4] and p.scale==[1.0]
   attn=pipe.unet.get_submodule(name.removesuffix('.processor'));heads=attn.heads;dim=p.to_k_ip[0].weight.shape[0]//heads
   for key,module in [('to_k_ip',p.to_k_ip[0]),('to_v_ip',p.to_v_ip[0])]:assert torch.equal(f.get_tensor(f'ip_adapter.{1+2*i}.{key}.weight').half(),module.weight.detach().cpu())
   bank[f'l{i:02d}']=np.empty((4,4,heads,2,dim),np.float16);layouts.append(dict(index=i,name=name,ip_index=1+2*i,heads=heads,head_dim=dim))
   handles.append(p.to_v_ip[0].register_forward_hook(hook(i)))
 import diffusers.schedulers.scheduling_euler_ancestral_discrete as EA
 old_noise,old_prepare=EA.randn_tensor,pipe.prepare_latents;noise={}
 def cn(*args,**kwargs):
  z=old_noise(*args,**kwargs);noise['ancestral'].append(tensor_sha(z));return z
 def ci(*args,**kwargs):
  z=old_prepare(*args,**kwargs);noise['initial'].append(tensor_sha(z));return z
 EA.randn_tensor,pipe.prepare_latents=cn,ci
 def run(i):
  global noise
  noise={'initial':[],'ancestral':[]}
  with torch.inference_mode():z=pipe(prompt_embeds=prompt,pooled_prompt_embeds=pooled,height=512,width=512,ip_adapter_image_embeds=[torch.tensor(values[i,0][None,None],device='cuda').half()],num_inference_steps=4,guidance_scale=0.,generator=torch.Generator(device='cuda').manual_seed(1729+int(ids[i])),output_type='latent').images[0]
  assert len(noise['initial'])==1 and len(noise['ancestral'])==4
  return tensor_sha(z),dict(noise)
 baseline=run(0);F.scaled_dot_product_attention=sdpa
 for current,i in enumerate(q['Q_probe_indices']):
  guard();counts=[0]*70;active=True;actual=run(i);active=False
  assert counts==[4]*70 and pending is None
  if i==0:assert actual==baseline
  proofs.append(dict(probe_index=i,T_row=int(ids[i]),seed=1729+int(ids[i]),final_latent_sha256=actual[0],noise=actual[1],units=280))
 F.scaled_dot_product_attention=original
 for h in handles:h.remove()
 handles=[]
 assert frozen=={n:tensor_digest(v,half=True) for n,v in [('UNet',pipe.unet),('VAE',pipe.vae)]}
 assert all(np.isfinite(v).all() for v in bank.values());np.savez_compressed(out/'Q.npz',**bank);guard()
 K.write(out/'completion.json',dict(status='PASS_FOUR_T_PROBES_ALL70_LAYERS_FOUR_TIMES_EXACT_NATIVE_Q',bank=K.ref(out/'Q.npz'),layouts=layouts,trajectories=proofs,all140_KV_weights_exact=True,T0_hook_final_latent_and_all_noises_bitexact=True,host_weights_unchanged=True,UNet_NFE=20,new_research_images=0,loader=loader,seconds=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0))
except BaseException as exc:K.write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
finally:
 F.scaled_dot_product_attention=original
 for h in handles:h.remove()
 if 'EA' in globals():EA.randn_tensor,pipe.prepare_latents=old_noise,old_prepare
