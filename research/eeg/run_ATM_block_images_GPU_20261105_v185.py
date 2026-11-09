"""Unchanged native h94/Turbo images for the two fixed EEG supervision fits."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import argparse,fcntl,hashlib,io,json,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--mode',choices=['profile','generate','encode','paper'],required=True)
ap.add_argument('--domain',choices=['R4','R80'],default='R4');ap.add_argument('--start',type=int,default=0);ap.add_argument('--arm',default='');a=ap.parse_args()
import atm_block_image_contract_20261105_v185 as C
start=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;c=C.setup(a.policy);q,K,out=c['q'],c['K'],c['out']
from atm_repeat_image_clean_cache_20261105_v2 import release
K.write(out/('cache_advice_GPU_'+str(os.getpid())+'.json'),release(c['physical']['physical_source']))
before=C.resources(c,True)
assert K.ref(__file__)==q['sources']['GPU']
suffix=a.mode+('_'+a.domain+'_'+str(a.start) if a.mode=='generate' else '_'+a.domain if a.mode=='encode' else '_'+a.arm if a.mode=='paper' else '')
job=K.read(c['repo']/'audit'/(q['job_prefix']+'_'+suffix+'_job.json'));assert job['pid']==os.getpid() and job['status']=='RUNNING'
lock=(c['repo']/'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==c['parent']['GPU_UUID']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
import torch
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(c['parent']['maximum_GPU_bytes']/torch.cuda.get_device_properties(0).total_memory)
from atm_native_training_runtime_20261103_v1 import tensor_sha
def save_array(path,value):
 p=Path(path);tmp=p.with_name(p.name+'.pending');assert not p.exists() and not tmp.exists()
 with tmp.open('xb') as stream:np.save(stream,value,allow_pickle=False)
 tmp.replace(p)

def guard():
 assert time.monotonic()-start<850
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<c['parent']['maximum_extra_CPU_bytes']
 assert torch.cuda.max_memory_allocated()<c['parent']['maximum_GPU_bytes']
 used=K.allocated(q['project_root']);assert used<q['maximum_output_bytes'];s=os.statvfs(q['project_root'])
 assert s.f_bavail*s.f_frsize>=before['required_including_this_project']+before['own_used_bytes']-used
try:
 if a.mode in ('profile','generate'):
  from atm_native_training_runtime_20261103_v1 import load_host
  from brainhive_native_direction_contract_20261103_v1 import tensor_digest
  directory=out/('loader_'+suffix);directory.mkdir()
  pipe,projection,prompt,pooled,timeids,diffusion,proof=load_host(c['cv']['bq'],directory)
  import transformers
  assert transformers.__version__=='4.36.0'
  import diffusers.schedulers.scheduling_euler_ancestral_discrete as EA
  old_noise,old_prepare=EA.randn_tensor,pipe.prepare_latents;capture={};frozen={k:tensor_digest(v,half=True) for k,v in [('UNet',pipe.unet),('VAE',pipe.vae)]}
  def cn(*args,**kw):
   v=old_noise(*args,**kw);capture['ancestral'].append(tensor_sha(v));return v
  def ci(*args,**kw):
   v=old_prepare(*args,**kw);capture['initial'].append(tensor_sha(v));return v
  EA.randn_tensor,pipe.prepare_latents=cn,ci
  def generate(value,seed):
   vv=torch.tensor(value[None].copy(),device='cuda')
   with torch.inference_mode():
    token=projection(vv.half());im=pipe(prompt_embeds=prompt,pooled_prompt_embeds=pooled,height=512,width=512,ip_adapter_image_embeds=[vv[:,None].half()],num_inference_steps=4,guidance_scale=0.,generator=torch.Generator(device='cuda').manual_seed(seed)).images[0]
   return vv,token,im
  try:
   if a.mode=='profile':
    ref=c['positive'][0];capture={'initial':[],'ancestral':[]};v,t,im=generate(c['values'][0,0],ref['image_seed'])
    px=hashlib.sha256(np.asarray(im.convert('RGB')).tobytes()).hexdigest()
    K.write(out/'native_replay_actual.json',dict(pixels=px,expected=ref['pixels_sha256'],noise=capture,expected_noise=ref['noise'],tokens=tensor_sha(t),expected_tokens=ref['tokens_FP16_sha256']))
    assert px==ref['pixels_sha256'] and capture==ref['noise'] and tensor_sha(t)==ref['tokens_FP16_sha256']
    im.save(out/'frozen_positive_R4_replay.webp',format='WEBP',lossless=True,quality=100)
    tokens=np.empty((600,4,4,2048),np.float16)
    with torch.inference_mode():
     for i in range(600):
      if i%32==0:guard()
      for j in range(4):tokens[i,j]=projection(torch.tensor(c['values'][i,j][None].copy(),device='cuda').half()).cpu().numpy()[0]
    assert np.isfinite(tokens).all()
    for domain in ('R4','R80'):
     offset=0 if domain=='R4' else 400
     for j,arm in enumerate(C.ARMS[:2]):
      for i,rr in enumerate(c['reference_records'][domain][arm]):
       vv=c['values'][offset+i,j][None];tt=tokens[offset+i,j][None]
       assert hashlib.sha256(vv.tobytes()).hexdigest()==rr['condition_FP32_sha256']
       assert hashlib.sha256(tt.tobytes()).hexdigest()==rr['tokens_FP16_sha256']
    save_array(out/'tokens.npy',tokens);save_array(out/'conditions.npy',c['values'])
    for arm in C.ARMS[:2]:
     prior=K.read(c['reference']['out']/('paper_'+arm+'_completion.json'));K.checked(prior['values'])
     K.write(out/('paper_'+arm+'_completion.json'),dict(prior,reused_reference=True,reference_completion=K.ref(c['reference']['out']/('paper_'+arm+'_completion.json'))))
    result=dict(status='PASS_EXACT_FROZEN_POSITIVE_PIXEL_AND_ALL2400_TOKEN_PROFILE',tokens=K.ref(out/'tokens.npy'),conditions=K.ref(out/'conditions.npy'),
     baseline_R4_exact_historical_conditions_used=True,reexport_max_difference=c['baseline_reexport_max_difference'],engineering_images=1)
   else:
    profile=K.read(out/'profile_completion.json');assert profile['status']=='PASS_EXACT_FROZEN_POSITIVE_PIXEL_AND_ALL2400_TOKEN_PROFILE'
    tokens=np.load(K.checked(profile['tokens']));domain=a.domain;offset=0 if domain=='R4' else 400;n=400 if domain=='R4' else 200
    block=80 if domain=='R4' else 50;assert a.start in range(0,n,block)
    newarms=C.ARMS[2:]
    for arm in newarms:
     d=out/'images'/domain/arm;d.mkdir(parents=True,exist_ok=True);jp=d/'images.jsonl';old=[json.loads(s) for s in jp.read_text().splitlines()] if jp.exists() else []
     assert [r['query_index'] for r in old]==list(range(a.start))
    for i in range(a.start,a.start+block):
     guard();row=c['queries'][domain][i];expected_noise=c['positive'][i]['noise'] if domain=='R4' else c['r80noise'][i]['noise']
     for arm in newarms:
      j=C.ARMS.index(arm);capture={'initial':[],'ancestral':[]};v,t,im=generate(c['values'][offset+i,j],row['image_seed'])
      assert capture==expected_noise and np.array_equal(t.cpu().numpy()[0],tokens[offset+i,j])
      px=np.asarray(im.convert('RGB'));assert px.shape==(512,512,3);buf=io.BytesIO();im.save(buf,format='WEBP',lossless=True,quality=100)
      path=out/'images'/domain/arm/f'q{i:03d}.webp';assert not path.exists();assert K.allocated(q['project_root'])+len(buf.getvalue())+65536<q['maximum_output_bytes'];path.write_bytes(buf.getvalue())
      record={**row,'domain':domain,'query_index':i,'arm':arm,'image':K.ref(path),'pixels_sha256':hashlib.sha256(px.tobytes()).hexdigest(),'noise':capture,
       'condition_FP32_sha256':tensor_sha(v),'condition_FP16_sha256':tensor_sha(v.half()),'tokens_FP16_sha256':tensor_sha(t),'condition_bank':profile['conditions'],
       'normal_inference_only_EEG':True,'query_target_or_concept_network_input':False,'UNet_NFE':4,'prior_NFE':0,'EEG_deployment_forward_calls':1}
      with (path.parent/'images.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
     K.write(out/'generation_progress.json',dict(domain=domain,complete_queries_per_arm=i+1,new_arms=list(newarms),effects_read=False))
    result=dict(status='COMPLETE_COUNT_RISK_IMAGE_BLOCK',domain=domain,start=a.start,queries=block,new_images=block*len(newarms))
  finally:EA.randn_tensor,pipe.prepare_latents=old_noise,old_prepare
  assert frozen=={k:tensor_digest(v,half=True) for k,v in [('UNet',pipe.unet),('VAE',pipe.vae)]};result['loader']=proof
 elif a.mode=='encode':
  audit=K.read(out/'image_audit.json');assert audit['status']=='PASS_ALL2400_IMAGE_PROVENANCE'
  import image_evaluation as E
  assert K.ref(E.__file__)==q['runtime_sources']['image_evaluation'] and E.DINO_SHA256==q['DINO_weight_sha256']
  model=E.dino_model();n=400 if a.domain=='R4' else 200;features=np.empty((n,4,768),np.float32);records=C.records(c,a.domain)
  previous=K.read(c['reference']['out']/('encode_'+a.domain+'_completion.json'));refvalues=np.load(K.checked(previous['features']));target=np.load(K.checked(previous['target']))
  features[:,:2]=refvalues[:,[0,2]];error=0.
  for j,arm in enumerate(C.ARMS[2:],start=2):
   for lo in range(0,n,32):
    guard();batch=records[arm][lo:lo+32];features[lo:lo+len(batch),j]=E.dino_features([K.checked(r['image']) for r in batch],model,batch_size=32,real_images=False).float().numpy()
  assert features.shape==(n,4,768) and target.shape==(n,768) and np.isfinite(features).all() and np.isfinite(target).all()
  assert np.allclose(np.linalg.norm(features,axis=2),1,atol=3e-5)
  save_array(out/('features_'+a.domain+'.npy'),features);save_array(out/('targets_'+a.domain+'.npy'),target)
  result=dict(status='COMPLETE_FROZEN_DINO_FULL_DOMAIN',domain=a.domain,features=K.ref(out/('features_'+a.domain+'.npy')),target=K.ref(out/('targets_'+a.domain+'.npy')),historical_replay_max_error=None,historical_feature_extraction_reused=True,reference_features=previous['features'])
 else:
  assert a.arm in C.ARMS and K.read(out/'image_audit.json')['status']=='PASS_ALL2400_IMAGE_PROVENANCE'
  from brainhive_original_metrics_runtime_v1 import OriginalEvaluator,NAMES,transforms
  from PIL import Image
  assert tuple(NAMES)==C.PAPER;pre=transforms.Compose([transforms.Resize(512,interpolation=transforms.InterpolationMode.BILINEAR),transforms.CenterCrop(512),transforms.ToTensor()])
  rows=C.records(c,'R80')[a.arm];originals=[];generated=[]
  for r in rows:
   guard()
   with Image.open(K.checked(r['target'])) as im:originals.append(pre(im.convert('RGB')))
   with Image.open(K.checked(r['image'])) as im:generated.append(pre(im.convert('RGB')))
  ev=OriginalEvaluator()
  try:
   aggregate,per=ev.evaluate(torch.stack(originals),torch.stack(generated));values=np.stack([per[k] for k in C.PAPER],axis=1).astype(np.float64)
   save_array(out/('paper_'+a.arm+'.npy'),values);result=dict(status='COMPLETE_ALL200_AUTHOR_EIGHT_METRICS',arm=a.arm,values=K.ref(out/('paper_'+a.arm+'.npy')),metrics=list(C.PAPER),defined_counts=np.isfinite(values).sum(0).tolist(),numerical_source_checks=ev.checks)
  finally:ev.close()
 guard();result.update(policy=K.ref(a.policy),seconds=time.monotonic()-start,peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0)
 K.write(out/(suffix+'_completion.json'),result)
except BaseException as exc:
 K.write(out/(suffix+'_failure.json'),dict(error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-start));raise
