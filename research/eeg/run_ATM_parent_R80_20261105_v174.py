"""No fitting: exact-parent EEG inference, native images, full paired evaluation."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import argparse,fcntl,hashlib,io,json,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image
import atm_parent_R80_contract_20261105_v174 as C
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--mode',required=True,choices=['conditions','CPU_conditions','profile','generate','audit','encode','paper','score']);ap.add_argument('--start',type=int,default=0);a=ap.parse_args()
started=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;c=C.setup(a.policy);q,K,out=c['q'],c['K'],c['out'];assert K.ref(__file__)==q['sources']['runner']
from atm_repeat_image_clean_cache_20261105_v2 import release
release(c['physical']['physical_source']);before=C.resources(c,True)
gpu=a.mode in ('conditions','profile','generate','encode','paper');suffix=a.mode+('_'+str(a.start) if a.mode=='generate' else '')
if gpu:
 job=K.read(c['repo']/'audit'/(q['job_prefix']+'_'+suffix+'_job.json'));assert job['pid']==os.getpid() and job['status']=='RUNNING'
 lock=(c['repo']/'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==c['parent']['GPU_UUID']
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
import torch
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
if gpu:torch.cuda.set_per_process_memory_fraction(c['parent']['maximum_GPU_bytes']/torch.cuda.get_device_properties(0).total_memory)
from atm_native_training_runtime_20261103_v1 import tensor_sha
def guard():
 assert time.monotonic()-started<850 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<c['parent']['maximum_extra_CPU_bytes']
 if gpu:assert torch.cuda.max_memory_allocated()<c['parent']['maximum_GPU_bytes']
 used=K.allocated(q['project_root']);assert used<q['maximum_output_bytes'];st=os.statvfs(q['project_root'])
 assert st.f_bavail*st.f_frsize>=before['required_including_this_project']+before['own_used_bytes']-used
try:
 if a.mode in ('conditions','CPU_conditions'):
  sys.path[:0]=c['physical']['existing_runtime_paths']+['/root/atm_official_20261016']
  from models.atms import ATMS
  test=torch.load(K.checked(c['ref']['r80']['q']['EEG_R80']),map_location='cpu',weights_only=True,mmap=True);assert test.shape==(200,63,250)
  original=torch.load(K.checked(q['encoder']),map_location='cpu',weights_only=True,mmap=True)
  device='cuda' if gpu else 'cpu';net=ATMS().to(device).eval();net.load_state_dict(original,strict=True);assert sum(p.numel() for p in net.parameters())==3200413
  pred=np.empty((200,1024),np.float32)
  with torch.inference_mode():
   for lo in range(0,200,100):guard();pred[lo:lo+100]=net(test[lo:lo+100].to(device),torch.ones(100,device=device,dtype=torch.long)).cpu().numpy()
  assert all(torch.equal(v.cpu(),original[k]) for k,v in net.state_dict().items()) and np.isfinite(pred).all()
  if gpu:
   positive=c['ref']['values'][400:,0];assert positive.shape==(200,1024)
   C.save(out/'conditions.npy',np.stack([pred,positive],1));K.write(out/'queries.json',c['rows'])
   result=dict(status='COMPLETE_EXACT_PARENT_ALL200_R80_EEG_CONDITIONS',conditions=K.ref(out/'conditions.npy'),queries=K.ref(out/'queries.json'),new_updates=0)
  else:
   expected=np.load(K.checked(K.read(out/'conditions_completion.json')['conditions']))[:,0];errors=np.max(abs(pred-expected),axis=1);assert errors.max()<5e-5
   C.save(out/'independent_CPU_original_conditions.npy',pred);result=dict(status='PASS_ALL200_NATIVE_CPU_CONDITIONS',maximum_error=float(errors.max()),errors=errors.tolist(),all_original_weights_unchanged=True)
 elif a.mode in ('profile','generate'):
  assert K.read(out/'CPU_conditions_completion.json')['status']=='PASS_ALL200_NATIVE_CPU_CONDITIONS'
  cp=K.read(out/'conditions_completion.json');values=np.load(K.checked(cp['conditions']))
  from atm_native_training_runtime_20261103_v1 import load_host
  from brainhive_native_direction_contract_20261103_v1 import tensor_digest
  loader=out/('loader_'+suffix);loader.mkdir();pipe,projection,prompt,pooled,timeids,diffusion,proof=load_host(c['cv']['bq'],loader)
  import transformers
  assert transformers.__version__=='4.36.0'
  import diffusers.schedulers.scheduling_euler_ancestral_discrete as EA
  oldnoise,oldprepare=EA.randn_tensor,pipe.prepare_latents;capture={};frozen={k:tensor_digest(v,half=True) for k,v in [('UNet',pipe.unet),('VAE',pipe.vae)]}
  def cn(*args,**kw):
   v=oldnoise(*args,**kw);capture['ancestral'].append(tensor_sha(v));return v
  def ci(*args,**kw):
   v=oldprepare(*args,**kw);capture['initial'].append(tensor_sha(v));return v
  EA.randn_tensor,pipe.prepare_latents=cn,ci
  def render(value,seed):
   v=torch.tensor(value[None].copy(),device='cuda')
   with torch.inference_mode():
    t=projection(v.half());im=pipe(prompt_embeds=prompt,pooled_prompt_embeds=pooled,height=512,width=512,ip_adapter_image_embeds=[v[:,None].half()],num_inference_steps=4,guidance_scale=0.,generator=torch.Generator(device='cuda').manual_seed(seed)).images[0]
   return v,t,im
  try:
   if a.mode=='profile':
    ref=c['positive'][0];capture={'initial':[],'ancestral':[]};v,t,im=render(values[0,1],ref['image_seed']);px=hashlib.sha256(np.asarray(im.convert('RGB')).tobytes()).hexdigest()
    K.write(out/'native_replay_actual.json',dict(pixels=px,expected=ref['pixels_sha256'],noise=capture,expected_noise=ref['noise'],tokens=tensor_sha(t),expected_tokens=ref['tokens_FP16_sha256']))
    assert px==ref['pixels_sha256'] and capture==ref['noise'] and tensor_sha(t)==ref['tokens_FP16_sha256']
    im.save(out/'positive_R80_replay.webp',format='WEBP',lossless=True,quality=100)
    tokens=np.empty((200,2,4,2048),np.float16)
    with torch.inference_mode():
     for i in range(200):
      if i%32==0:guard()
      for j in range(2):tokens[i,j]=projection(torch.tensor(values[i,j][None].copy(),device='cuda').half()).cpu().numpy()[0]
      assert C.sha(tokens[i,1])==c['positive'][i]['tokens_FP16_sha256'] and C.sha(values[i,1])==c['positive'][i]['condition_FP32_sha256']
    assert np.isfinite(tokens).all();C.save(out/'tokens.npy',tokens);result=dict(status='PASS_EXACT_R80_NCE_PIXEL_AND_ALL400_TOKENS',tokens=K.ref(out/'tokens.npy'),conditions=cp['conditions'])
   else:
    profile=K.read(out/'profile_completion.json');assert profile['status']=='PASS_EXACT_R80_NCE_PIXEL_AND_ALL400_TOKENS';tokens=np.load(K.checked(profile['tokens']));assert a.start in (0,100)
    directory=out/'images';directory.mkdir(exist_ok=True);jp=directory/'images.jsonl';previous=[json.loads(s) for s in jp.read_text().splitlines()] if jp.exists() else [];assert [r['query_index'] for r in previous]==list(range(a.start))
    for i in range(a.start,a.start+100):
     guard();r=c['rows'][i];capture={'initial':[],'ancestral':[]};v,t,im=render(values[i,0],r['image_seed'])
     assert capture==c['positive'][i]['noise'] and np.array_equal(t.cpu().numpy()[0],tokens[i,0])
     px=np.asarray(im.convert('RGB'));assert px.shape==(512,512,3);p=directory/f'q{i:03d}.webp';assert not p.exists();buf=io.BytesIO();im.save(buf,format='WEBP',lossless=True,quality=100)
     assert K.allocated(q['project_root'])+len(buf.getvalue())+65536<q['maximum_output_bytes'];p.write_bytes(buf.getvalue())
     record={**r,'query_index':i,'arm':'original_ATMS','image':K.ref(p),'pixels_sha256':hashlib.sha256(px.tobytes()).hexdigest(),'noise':capture,
      'condition_FP32_sha256':tensor_sha(v),'condition_FP16_sha256':tensor_sha(v.half()),'tokens_FP16_sha256':tensor_sha(t),'condition_bank':cp['conditions'],
      'encoder':q['encoder'],'normal_inference_only_EEG':True,'query_target_or_concept_network_input':False,'UNet_NFE':4,'prior_NFE':0}
     with jp.open('a') as f:f.write(json.dumps(record)+'\n')
     K.write(out/'generation_progress.json',dict(complete=i+1,total=200,effects_read=False))
    result=dict(status='COMPLETE_ORIGINAL_R80_BLOCK',start=a.start,new_images=100)
  finally:EA.randn_tensor,pipe.prepare_latents=oldnoise,oldprepare
  assert frozen=={k:tensor_digest(v,half=True) for k,v in [('UNet',pipe.unet),('VAE',pipe.vae)]};result['loader']=proof
 elif a.mode=='audit':
  pr=K.read(out/'profile_completion.json');values=np.load(K.checked(pr['conditions']));tokens=np.load(K.checked(pr['tokens']));rr=C.records(c);paths=set();inodes=set();ordered=[];dups={};near=[]
  for i,row in enumerate(c['rows']):
   guard();K.checked(row['target'])
   for j,arm in enumerate(C.ARMS):
    r=rr[arm][i];assert all(r[k]==row[k] for k in C.IDENTITY) and r['target']==row['target'] and r['arm']==arm
    p=K.checked(r['image']);st=p.stat();inode=(st.st_dev,st.st_ino);assert str(p) not in paths and inode not in inodes;paths.add(str(p));inodes.add(inode)
    with Image.open(p) as im:rgb=np.asarray(im.convert('RGB'))
    assert rgb.shape==(512,512,3) and C.sha(rgb)==r['pixels_sha256']
    assert C.sha(values[i,j])==r['condition_FP32_sha256'] and C.sha(values[i,j].astype(np.float16))==r['condition_FP16_sha256'] and C.sha(tokens[i,j])==r['tokens_FP16_sha256']
    assert r['noise']==c['positive'][i]['noise'] and len(r['noise']['initial'])==1 and len(r['noise']['ancestral'])==4
    if j==0:assert r['condition_bank']==pr['conditions'] and r['encoder']==q['encoder'] and p.is_relative_to(out/'images') and r['normal_inference_only_EEG'] and not r['query_target_or_concept_network_input'] and r['UNet_NFE']==4
    dups.setdefault(r['pixels_sha256'],[]).append([arm,i])
    if np.max(rgb.astype(float).std((0,1)))<2.55:near.append([arm,i])
    ordered.append(r)
  assert len(ordered)==400;K.write(out/'ordered_images.json',ordered);result=dict(status='PASS_ALL400_R80_IMAGES_CONDITIONS_NOISE_PIXELS',ordered_records=K.ref(out/'ordered_images.json'),near_constant=near,duplicates=[v for v in dups.values() if len(v)>1],new_images=200,reused_images=200)
 elif a.mode=='encode':
  assert K.read(out/'audit_completion.json')['status']=='PASS_ALL400_R80_IMAGES_CONDITIONS_NOISE_PIXELS'
  import image_evaluation as E
  assert K.ref(E.__file__)==q['runtime_sources']['image_evaluation'] and E.DINO_SHA256==q['DINO_weight_sha256'];model=E.dino_model();rr=C.records(c)
  features=np.empty((200,2,768),np.float32)
  for j,arm in enumerate(C.ARMS):
   for lo in range(0,200,32):guard();batch=rr[arm][lo:lo+32];features[lo:lo+len(batch),j]=E.dino_features([K.checked(r['image']) for r in batch],model,batch_size=32,real_images=False).float().numpy()
  prior=K.read(c['ref']['out']/'encode_R80_completion.json');pf=np.load(K.checked(prior['features']))[:,0];error=float(abs(features[:,1]-pf).max());assert error<3e-6;features[:,1]=pf
  targets=np.load(K.checked(prior['target']));assert targets.shape==(200,768) and np.isfinite(features).all();C.save(out/'features.npy',features);C.save(out/'targets.npy',targets)
  result=dict(status='COMPLETE_ALL400_DINO_FEATURES',features=K.ref(out/'features.npy'),targets=K.ref(out/'targets.npy'),reused_feature_replay_max_error=error)
 elif a.mode=='paper':
  assert K.read(out/'audit_completion.json')['status']=='PASS_ALL400_R80_IMAGES_CONDITIONS_NOISE_PIXELS'
  from brainhive_original_metrics_runtime_v1 import OriginalEvaluator,NAMES,transforms
  assert tuple(NAMES)==C.PAPER;pre=transforms.Compose([transforms.Resize(512,interpolation=transforms.InterpolationMode.BILINEAR),transforms.CenterCrop(512),transforms.ToTensor()]);real=[];gen=[]
  for r in C.records(c)['original_ATMS']:
   guard()
   with Image.open(K.checked(r['target'])) as im:real.append(pre(im.convert('RGB')))
   with Image.open(K.checked(r['image'])) as im:gen.append(pre(im.convert('RGB')))
  ev=OriginalEvaluator()
  try:
   aggregate,per=ev.evaluate(torch.stack(real),torch.stack(gen));data=np.stack([per[k] for k in C.PAPER],1).astype(np.float64);C.save(out/'paper_original.npy',data)
   result=dict(status='COMPLETE_ORIGINAL200_AUTHOR_EIGHT_METRICS',values=K.ref(out/'paper_original.npy'),numerical_source_checks=ev.checks)
  finally:ev.close()
 else:
  assert K.read(out/'audit_completion.json')['status']=='PASS_ALL400_R80_IMAGES_CONDITIONS_NOISE_PIXELS'
  import enigma_R80_count_image_contract_20261104_v1 as R
  import score_low_level as L
  assert K.ref(L.__file__)==q['runtime_sources']['low_level'];enc=K.read(out/'encode_completion.json');feat=np.load(K.checked(enc['features']));target=np.load(K.checked(enc['targets']));rr=C.records(c);arrays={};matrices={}
  ownpaper=np.load(K.checked(K.read(out/'paper_completion.json')['values']));oldpaper=np.load(K.checked(K.read(c['ref']['out']/'paper_frozen_positive_NCE_completion.json')['values']));assert ownpaper.shape==oldpaper.shape==(200,8)
  check,_=R.metrics(target,target);assert np.all(check['DINO2AFC']==1) and np.all(check['gallery_top1']==1)
  for j,arm in enumerate(C.ARMS):
   arrays[arm],matrices[arm]=R.metrics(feat[:,j],target)
   for k in ('PixCorr_RGB','SSIM_Y','mean_RGB_absolute_error'):arrays[arm][k]=np.empty(200)
   for k,name in enumerate(C.PAPER):arrays[arm][name]=(ownpaper if j==0 else oldpaper)[:,k]
  for i,r in enumerate(c['rows']):
   guard();trgb,ty=L.pixels(K.checked(r['target']));tm=L.moments(ty)
   for arm in C.ARMS:
    rgb,y=L.pixels(K.checked(rr[arm][i]['image']));arrays[arm]['PixCorr_RGB'][i]=L.pixcorr(rgb,trgb);arrays[arm]['SSIM_Y'][i]=L.ssim_y(y,ty,tm);arrays[arm]['mean_RGB_absolute_error'][i]=abs(rgb.mean((0,1))-trgb.mean((0,1))).mean()
  pairs,diffs=C.paired(arrays,c['rows'])
  for name,data in [('metrics.npz',{arm+'__'+k:v for arm,d in arrays.items() for k,v in d.items()}),('matrices.npz',matrices),('paired_differences.npz',diffs)]:
   with (out/name).open('xb') as f:np.savez_compressed(f,**data)
  result=dict(status='COMPLETE_EXACT_PARENT_R80_ALL15_PAIRED_METRICS',means={arm:{k:float(v.mean()) if np.isfinite(v).all() else None for k,v in d.items()} for arm,d in arrays.items()},
   positive_NCE_minus_original=pairs,metrics=K.ref(out/'metrics.npz'),matrices=K.ref(out/'matrices.npz'),differences=K.ref(out/'paired_differences.npz'),
   all400_before_scores=True,single_subject=True,physical_R=80,training_T_concepts=1323,bootstrap_seed=1729,bootstrap_replicates=1000,
   explanatory_limit='Exposed benchmark, no independent confirmation; continued NCE is prior art; checkpoint/training settings differ. No attribution solely to loss.',
   original_encoder_V_selection_history=True,train_only_MVNN_not_reestablished=True,not_ten_subject_SOTA=True,lower_better=['mean_RGB_absolute_error','eval_effnet','eval_swav'])
 guard();result.update(seconds=time.monotonic()-started,extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0,peak_GPU_bytes=torch.cuda.max_memory_allocated() if gpu else 0)
 K.write(out/(suffix+'_completion.json'),result)
except BaseException as exc:
 K.write(out/(suffix+'_failure.json'),dict(error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-started));raise
