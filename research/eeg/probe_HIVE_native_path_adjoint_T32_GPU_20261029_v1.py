"""Frozen T-only comparison of pointwise training and actual four-step gradients.

This is a local input-gradient diagnostic, not a new fitted EEG method.  The
endpoint is the native VAE latent, not semantic image quality.  Autograd through
FP16 casts is a continuous relaxation, not the derivative of quantized files.
"""
import argparse
import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback


def ref(p):
 p=Path(p)
 with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),bytes=p.stat().st_size,sha256=digest)


def checked(item):
 assert ref(item['path'])==item
 return Path(item['path'])


def write(p,value):
 with Path(p).open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def memory():
 return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024


def run(path,lo,hi):
 s=json.loads(Path(path).read_bytes());assert s['host']=='old' and (lo,hi) in ((0,16),(16,32))
 for item in s['source_pins']:checked(item)
 assert 'run_gpu_job_v2.py' in (Path('/proc')/str(os.getppid())/'cmdline').read_text()
 assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==s['GPU_UUID']
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 root=Path(s['output_root']);out=root/f'segment_{lo:02d}_{hi:02d}';out.mkdir();began=time.monotonic();phase='RUNTIME';base=memory()
 sys.path.insert(0,s['repository_root'])
 import brainhive_paired_study_contract_20261025_v1 as C
 C.setup_runtime(dict(runtime_manifest=s['runtime_manifest']))
 import numpy as np
 import torch
 from torch.utils.checkpoint import checkpoint
 from diffusers import DDPMScheduler
 from brainhive_bounded_component_v1 import BoundedConditionAdapter
 from brainhive_component_generator_v1 import load_native,denoise
 from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
 torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 stop=threading.Event()
 def guard():
  while not stop.wait(.25):
   gpu=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
   if time.monotonic()-began>800 or memory()-base>10<<30 or gpu>20<<30:
    write(out/'resource_limit.json',dict(seconds=time.monotonic()-began,extra_CPU=memory()-base,GPU_allocated=gpu));os.kill(os.getpid(),signal.SIGTERM);return
 monitor=threading.Thread(target=guard,daemon=True);monitor.start()
 try:
  bank=np.load(checked(s['input_bank']),allow_pickle=False);assert bank['condition'].shape==(32,1024) and bank['latent'].shape==(32,4,64,64)
  state=torch.load(checked(s['ordinary']),weights_only=True,map_location='cpu');assert state['calibration']['selected_strength']==1.
  ordinary=BoundedConditionAdapter(predict_risk=False).cuda().eval().requires_grad_(False);ordinary.load_state_dict(state['model'],strict=True)
  phase='EXACT_NATIVE_LOAD';(out/'loader').mkdir();observer=ConvertedWeightAudit(torch,out/'loader');observer.install()
  try:pipe,A,P,prompts,host=load_native(training=False)
  finally:observer.restore()
  loading=observer.complete();torch.cuda.set_per_process_memory_fraction(min(1.,float(20<<30)/torch.cuda.get_device_properties(0).total_memory))
  assert all(not m.training and all(not p.requires_grad for p in m.parameters()) for m in (pipe.unet,pipe.vae,A,P,ordinary))
  call=type(pipe).__call__;assert hasattr(call,'__wrapped__');native_grad_call=call.__wrapped__
  call_source=Path(inspect.getsourcefile(native_grad_call));assert ref(call_source)==s['native_pipeline_source']
  training_prompts=tuple(x.to(torch.bfloat16) for x in prompts)
  noise_scheduler=DDPMScheduler.from_config(pipe.scheduler.config);assert noise_scheduler.config.prediction_type=='epsilon'
  def fingerprint(module):
   h=hashlib.sha256()
   for name,tensor in module.state_dict().items():
    h.update(name.encode());h.update(str(tensor.dtype).encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
   return h.hexdigest()
  modules=dict(UNet=pipe.unet,VAE=pipe.vae,A=A,P=P,ordinary=ordinary)
  before={k:fingerprint(m) for k,m in modules.items()}
  original_forward=pipe.unet.forward;forward_counts={};forward_phase=['unset']
  def observed_forward(*args,**kwargs):
   forward_counts[forward_phase[0]]=forward_counts.get(forward_phase[0],0)+1
   return original_forward(*args,**kwargs)
  pipe.unet.forward=observed_forward
  def checkpointed_forward(*args,**kwargs):
   return checkpoint(observed_forward,*args,use_reentrant=False,**kwargs)
  def sample(tokens,seed,grad):
   observations=[];step_module=inspect.getmodule(type(pipe.scheduler));original_randn=step_module.randn_tensor
   def observed_randn(*args,**kwargs):
    x=original_randn(*args,**kwargs);observations.append(hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest());return x
   step_module.randn_tensor=observed_randn
   kwargs=dict(height=512,width=512,generator=torch.Generator(device='cuda').manual_seed(seed),prompt_embeds=prompts[0],pooled_prompt_embeds=prompts[1],cross_attention_kwargs=dict(ip_hidden_states=tokens),num_inference_steps=4,guidance_scale=0.,output_type='latent')
   if grad:pipe.unet.forward=checkpointed_forward
   try:
    if grad:result=native_grad_call(pipe,**kwargs).images
    else:
     with torch.no_grad():result=pipe(**kwargs).images
   finally:pipe.unet.forward=observed_forward;step_module.randn_tensor=original_randn
   assert len(observations)==4 and result.shape==(1,4,64,64)
   return result,observations
  records=[];gradients=[];phase='FIXED_T32_FOUR_STEP_AND_POINT_GRADIENTS'
  for index in range(lo,hi):
   q=s['queries'][index];c=torch.from_numpy(bank['condition'][index:index+1].copy()).cuda();w=torch.from_numpy(bank['gamma'][index:index+1].copy()).cuda();b=torch.from_numpy(bank['beta'][index:index+1].copy()).cuda();real=torch.from_numpy(bank['latent'][index:index+1].copy()).cuda();teacher=torch.from_numpy(bank['target'][index:index+1].copy()).cuda()
   forward_phase[0]='native_replay';counts_before=dict(forward_counts)
   with torch.no_grad():
    fixed=ordinary(c,w,b)[0].detach().clone();native_tokens=P(A(fixed.half()));native,noises=sample(native_tokens,q['image_seed'],False)
   forward_phase[0]='path_forward'
   delta=torch.zeros_like(fixed,requires_grad=True);tokens=P(A((fixed+w*delta).half()));actual,grad_noises=sample(tokens,q['image_seed'],True)
   assert noises==grad_noises and torch.equal(native,actual.detach()),'Gradient-enabled path must exactly preserve all four native steps and random noise'
   end_loss=torch.nn.functional.mse_loss(actual.float(),real)
   forward_phase[0]='path_backward_recompute';g_end=torch.autograd.grad(end_loss,delta)[0].detach().float().cpu().numpy()[0]
   endpoint=actual.detach().cpu().numpy();del delta,tokens,actual
   delta=torch.zeros_like(fixed,requires_grad=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):
    tok=P(A(fixed+w*delta))
    with torch.no_grad():teacher_tok=P(A(teacher))
    token_loss=(1-torch.nn.functional.cosine_similarity(tok.float(),teacher_tok.float(),dim=-1)).mean()
   g_token=torch.autograd.grad(token_loss,delta)[0].detach().float().cpu().numpy()[0]
   g_points=[];losses=[];times=[];rng=torch.Generator(device='cuda').manual_seed(q['point_noise_seed'])
   forward_phase[0]='point_forward'
   for draw in range(4):
    t=torch.randint(0,1000,(1,),device='cuda',generator=rng);noise=torch.randn(real.shape,device='cuda',generator=rng);noisy=noise_scheduler.add_noise(real,noise,t);delta=torch.zeros_like(fixed,requires_grad=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
     tok=P(A(fixed+w*delta));prediction=denoise(pipe,training_prompts,noisy,t,tok);loss=torch.nn.functional.mse_loss(prediction.float(),noise)
    g_points.append(torch.autograd.grad(loss,delta)[0].detach().float().cpu().numpy()[0]);losses.append(float(loss.detach()));times.append(int(t.item()));del delta,tok,prediction,loss
   u=((fixed-b)/w).detach().float().cpu().numpy()[0];all_grad=np.stack([g_end,g_token]+g_points);assert np.isfinite(all_grad).all() and np.isfinite(endpoint).all()
   archive=out/f'query_{index:02d}.npz'
   with archive.open('xb') as f:np.savez(f,u=u,grad_end=g_end,grad_token=g_token,grad_epsilon=np.stack(g_points),native_endpoint=endpoint,real_latent=bank['latent'][index:index+1])
   counts={k:v-counts_before.get(k,0) for k,v in forward_counts.items()};assert counts['native_replay']==counts['path_forward']==counts['point_forward']==4 and counts['path_backward_recompute']==4
   r=dict(index=index,query=q,archive=ref(archive),native_four_step_latent_bit_exact=True,ancestral_noise_hashes=noises,actual_UNet_forward_counts=counts,endpoint_latent_MSE=float(end_loss.detach()),token_angular_loss=float(token_loss.detach()),epsilon_MSE=losses,point_timesteps=times,seconds_since_segment_start=time.monotonic()-began)
   write(out/f'query_{index:02d}.json',r);records.append(r);gradients.append(all_grad)
   assert sum(p.stat().st_size for p in root.rglob('*') if p.is_file())<=s['maximum_output_bytes']
   del native,fixed,c,w,b,real,teacher,native_tokens,end_loss,token_loss;torch.cuda.empty_cache()
   print(json.dumps(dict(query=index,completed=len(records),expected=hi-lo,seconds=time.monotonic()-began)),flush=True)
  pipe.unet.forward=original_forward
  after={k:fingerprint(m) for k,m in modules.items()};assert before==after and ref(s['ordinary']['path'])==s['ordinary']
  assert all(all(p.grad is None and not p.requires_grad for p in m.parameters()) for m in modules.values())
  write(out/'completion.json',dict(status='COMPLETE_FROZEN_NATIVE_T32_PATH_GRADIENT_SEGMENT',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),range=[lo,hi],queries=hi-lo,records=records,protocol=ref(path),loader=loading,host=host,frozen_module_fingerprints_before=before,frozen_module_fingerprints_after=after,all_native_endpoints_bit_exact=True,actual_UNet_forward_counts=forward_counts,UNet_NFE_per_query=16,UNet_NFE_note='4 native replay +4 gradient forward +4 checkpoint recomputation +4 point forward; backward operations are additional compute.',new_model_updates=0,new_research_images=0,D_V_official_test_read=False,seconds=time.monotonic()-began,extra_CPU_peak=memory()-base,peak_GPU_allocated=torch.cuda.max_memory_allocated(),interpretation='Input gradients through the continuous FP16 relaxation. Endpoint latent MSE is not semantic image quality; point loss uses four fixed Monte Carlo draws from the original uniform-t objective.'))
 except BaseException as error:
  write(out/'failure.json',dict(status='T32_PATH_GRADIENT_PROBE_FAILED_PRESERVE_PREFIX_NO_AUTO_RETRY',phase=phase,error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-began));raise
 finally:stop.set();monitor.join(timeout=2)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--lo',type=int,required=True);p.add_argument('--hi',type=int,required=True);q=p.parse_args();run(q.protocol,q.lo,q.hi)
