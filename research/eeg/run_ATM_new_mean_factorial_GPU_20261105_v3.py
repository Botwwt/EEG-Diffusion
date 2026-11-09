"""Native four-step images under pinned within-concept token interventions."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import argparse,fcntl,hashlib,io,json,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import atm_new_mean_factorial_contract_20261105_v3 as C
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--mode',choices=['generate','encode'],required=True);ap.add_argument('--start',type=int,default=0);a=ap.parse_args()
started=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
c=C.setup(a.policy);K,q,out=c['K'],c['q'],c['out'];prior=c['prior'];parent=prior['parent']
assert K.ref(__file__)==q['sources']['GPU']
from atm_repeat_image_clean_cache_20261105_v2 import release
cache=release(prior['physical']['physical_source']);before=C.resources(c,starting=True)
suffix=a.mode+('_'+str(a.start) if a.mode=='generate' else '')
job=K.read(c['repo']/'audit'/(q['job_prefix']+'_'+suffix+'_job.json'));assert job['pid']==os.getpid() and job['status']=='RUNNING'
lock=(c['repo']/'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==parent['GPU_UUID']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
import torch
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(parent['maximum_GPU_bytes']/torch.cuda.get_device_properties(0).total_memory)
from atm_native_training_runtime_20261103_v1 import tensor_sha
from brainhive_native_direction_contract_20261103_v1 import tensor_digest
def guard():
 assert time.monotonic()-started<850
 used=K.allocated(q['project_root']);assert used<q['maximum_output_bytes']
 fs=os.statvfs(q['project_root']);assert fs.f_bavail*fs.f_frsize>=before['required_including_this_project']+before['own_used_bytes']-used
 assert torch.cuda.max_memory_allocated()<parent['maximum_GPU_bytes']
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<parent['maximum_extra_CPU_bytes']
try:
 if a.mode=='generate':
  assert a.start in (0,80,160,240,320)
  from atm_native_training_runtime_20261103_v1 import load_host
  ld=out/('loader_'+str(a.start)+'_recovery_v3');ld.mkdir()
  pipe,projection,prompt,pooled,ti,diffusion,proof=load_host(prior['cv']['bq'],ld)
  frozen=[tensor_digest(m,half=True) for m in (pipe.unet,pipe.vae,projection)]
  import diffusers.schedulers.scheduling_euler_ancestral_discrete as EA
  old_noise,old_prepare=EA.randn_tensor,pipe.prepare_latents;capture={};state=None
  def cn(*args,**kwargs):
   v=old_noise(*args,**kwargs);capture['ancestral'].append(tensor_sha(v));return v
  def ci(*args,**kwargs):
   v=old_prepare(*args,**kwargs);capture['initial'].append(tensor_sha(v));return v
  def hook(module,args,output):
   if state is None:return None
   i,j=state['i'],state['j'];actual=output.reshape(1,4,2048)
   assert torch.equal(actual,state['native'])
   if j==0:result=actual
   elif j==3:result=state['donor']
   else:
    x=actual.double();y=state['donor'].double();shift=y.mean(1,keepdim=True)-x.mean(1,keepdim=True)
    result=(x+shift if j==1 else y-shift).half()
   assert torch.equal(result,state['expected']);state['calls']+=1
   return result.reshape(output.shape)
  handle=projection.register_forward_hook(hook);EA.randn_tensor=cn;pipe.prepare_latents=ci
  def render(i,j):
   global state,capture
   d=int(c['donors'][i]);state=dict(i=i,j=j,native=torch.tensor(c['native'][i][None].copy(),device='cuda'),
    donor=torch.tensor(c['native'][d][None].copy(),device='cuda'),expected=torch.tensor(c['bank'][i,j][None].copy(),device='cuda'),calls=0)
   capture={'initial':[],'ancestral':[]}
   v=torch.tensor(c['values'][i][None].copy(),device='cuda')
   with torch.inference_mode():
    im=pipe(prompt_embeds=prompt,pooled_prompt_embeds=pooled,height=512,width=512,ip_adapter_image_embeds=[v[:,None].half()],
     num_inference_steps=4,guidance_scale=0.,generator=torch.Generator(device='cuda').manual_seed(c['rows'][i]['image_seed'])).images[0]
   assert state['calls']==4 and capture==prior['positive'][i]['noise'];state=None;return im
  try:
   if a.start==0 and not (out/'profile.json').exists():
    with torch.inference_mode():
     for i in range(400):
      if i%32==0:guard()
      p=projection(torch.tensor(c['values'][i][None].copy(),device='cuda').half()).reshape(4,2048)
      assert np.array_equal(p.cpu().numpy(),c['native'][i])
    im=render(0,0);assert C.sha(np.asarray(im.convert('RGB')))==prior['positive'][0]['pixels_sha256']
    im.save(out/'native_identity_replay.webp',format='WEBP',lossless=True,quality=100)
    K.write(out/'profile.json',dict(status='PASS_ALL400_NATIVE_TOKENS_AND_EXACT_IDENTITY_PIXEL_NOISE_REPLAY',native_tokens=q['inputs']['tokens'],all400_native_exact=True,identity_pixels_noise_exact=True))
   else:assert K.read(out/'profile.json')['status']=='PASS_ALL400_NATIVE_TOKENS_AND_EXACT_IDENTITY_PIXEL_NOISE_REPLAY'
   prefixes=[]
   for arm in C.ARMS[1:]:
    dr=out/'images'/arm;dr.mkdir(parents=True,exist_ok=True);p=dr/'images.jsonl'
    rows=[json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
    assert [r['query_index'] for r in rows]==list(range(len(rows))) and a.start<=len(rows)<=a.start+80
    prefixes.append(len(rows))
   assert len(set(prefixes))==1
   for i in range(prefixes[0],a.start+80):
    guard()
    if i%16==0:C.resources(c)
    for j,arm in enumerate(C.ARMS[1:],1):
     im=render(i,j);pixels=np.asarray(im.convert('RGB'));assert pixels.shape==(512,512,3)
     buff=io.BytesIO();im.save(buff,format='WEBP',lossless=True,quality=100);data=buff.getvalue()
     assert K.allocated(q['project_root'])+len(data)+65536<q['maximum_output_bytes']
     p=out/'images'/arm/f'q{i:03d}.webp';assert not p.exists();p.write_bytes(data)
     d=int(c['donors'][i]);row=dict(c['rows'][i],query_index=i,arm=arm,image=K.ref(p),pixels_sha256=C.sha(pixels),noise=capture,
      condition_FP32_sha256=C.sha(c['values'][i]),condition_FP16_sha256=C.sha(c['values'][i].astype(np.float16)),
      tokens_FP16_sha256=C.sha(c['bank'][i,j]),donor_query_index=d,donor_query_id=c['rows'][d]['query_id'],
      mean_donor_index=d if j in (1,3) else i,center_donor_index=d if j in (2,3) else i,
      donor_tokens_FP16_sha256=C.sha(c['native'][d]),UNet_NFE=4,projection_calls=4,
      privileged_same_concept_intervention=True,deployable_method=False,query_image_read_at_inference=False,
      frozen_cached_EEG_conditions=True,maximum_two_EEG_encodes_equivalent=True,policy=K.ref(a.policy))
     with (p.parent/'images.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    K.write(out/'generation_progress.json',dict(completed_queries_per_arm=i+1,block_start=a.start))
  finally:handle.remove();EA.randn_tensor=old_noise;pipe.prepare_latents=old_prepare
  assert frozen==[tensor_digest(m,half=True) for m in (pipe.unet,pipe.vae,projection)]
  result=dict(status='COMPLETE_240_NEW_MEAN_FACTORIAL_IMAGES',start=a.start,new_images=3*(a.start+80-prefixes[0]),preserved_prefix_per_arm=prefixes[0],all_frozen_weights_unchanged=True,loader=proof)
 else:
  assert K.read(out/'image_audit.json')['status']=='PASS_ALL1600_FACTORIAL_IMAGE_PROVENANCE'
  import image_evaluation as E
  assert K.ref(E.__file__)==prior['q']['runtime_sources']['image_evaluation'] and E.DINO_SHA256==prior['q']['DINO_weight_sha256']
  model=E.dino_model();rows=C.records(c);base=np.load(K.checked(q['inputs']['features']))[:,0]
  replay=E.dino_features([K.checked(rows['correct'][0]['image'])],model,batch_size=1,real_images=False).float().numpy()[0]
  error=float(abs(replay-base[0]).max());assert error<3e-6
  features=np.empty((400,4,768),np.float32);features[:,0]=base
  for j,arm in enumerate(C.ARMS[1:],1):
   for lo in range(0,400,32):
    guard();features[lo:lo+32,j]=E.dino_features([K.checked(r['image']) for r in rows[arm][lo:lo+32]],model,batch_size=32,real_images=False).float().numpy()
  assert np.isfinite(features).all() and np.allclose(np.linalg.norm(features,axis=2),1,atol=3e-5)
  C.save_array(out/'features.npy',features);result=dict(status='COMPLETE_FOUR_ARM_DINO_FEATURES',features=K.ref(out/'features.npy'),replay_max_error=error,new_images_encoded=1200,reused=400)
 guard();result.update(seconds=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0,
  cache_release=cache,resources=C.resources(c),source=K.ref(__file__))
 K.write(out/(suffix+'_completion.json'),result);print(json.dumps(result),flush=True)
except BaseException as e:
 K.write(out/(suffix+'_failure_recovery_v3.json'),dict(status='FAILED_PREFIX_RETAINED_NO_AUTO_RETRY',error=repr(e),traceback=traceback.format_exc()));raise
