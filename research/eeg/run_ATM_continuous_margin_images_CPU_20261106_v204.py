"""All 2,400 images, two galleries and every locked metric retained."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import argparse,hashlib,io,json,resource,sys,time,traceback,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import atm_continuous_margin_contract_20261106_v204 as C
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--mode',choices=['audit','score'],required=True);a=ap.parse_args()
start=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;c=C.setup(a.policy);q,K,out=c['q'],c['K'],c['out'];assert K.ref(__file__)==q['sources']['CPU'];C.resources(c)
def guard():
 assert time.monotonic()-start<850 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<c['parent']['maximum_extra_CPU_bytes']
try:
 profile=K.read(out/'profile_completion.json');assert profile['status']=='PASS_EXACT_FROZEN_POSITIVE_PIXEL_AND_ALL2400_TOKEN_PROFILE'
 tokens=np.load(K.checked(profile['tokens']));values=np.load(K.checked(profile['conditions']));assert np.array_equal(values,c['values'])
 records={d:C.records(c,d) for d in ['R4','R80']}
 if a.mode=='audit':
  assert not (out/'image_audit.json').exists();paths=set();inodes=set();ordered=[];duplicates={};near=[]
  metadata=K.read(K.checked(c['prepared']['target_metadata_ref']))['Dcheck'];assert len(metadata)==400
  for domain in ['R4','R80']:
   offset=0 if domain=='R4' else 400;n=400 if domain=='R4' else 200
   identity=('query_id','subject_id','stimulus_id','concept_id','array_row','image_seed')
   for i in range(n):
    guard();expected=c['queries'][domain][i];noise=c['positive'][i]['noise'] if domain=='R4' else c['r80noise'][i]['noise']
    if domain=='R4':assert all(metadata[i][k]==expected[k] for k in identity)
    else:K.checked(expected['target'])
    for j,arm in enumerate(C.ARMS):
     r=records[domain][arm][i];assert all(r[k]==expected[k] for k in identity) and r['arm']==arm
     p=K.checked(r['image']);st=p.stat();inode=(st.st_dev,st.st_ino);assert str(p) not in paths and inode not in inodes;paths.add(str(p));inodes.add(inode)
     with Image.open(p) as im:rgb=np.asarray(im.convert('RGB')).copy()
     assert rgb.shape==(512,512,3) and hashlib.sha256(rgb.tobytes()).hexdigest()==r['pixels_sha256']
     v=np.ascontiguousarray(values[offset+i,j][None]);t=np.ascontiguousarray(tokens[offset+i,j][None]);assert hashlib.sha256(v.tobytes()).hexdigest()==r['condition_FP32_sha256']
     assert hashlib.sha256(v.astype(np.float16).tobytes()).hexdigest()==r['condition_FP16_sha256'] and hashlib.sha256(t.tobytes()).hexdigest()==r['tokens_FP16_sha256']
     assert r['noise']==noise and len(noise['initial'])==1 and len(noise['ancestral'])==4
     if domain=='R80':assert r['target']==expected['target'] and str(p)!=expected['target']['path']
     if j>=2:
      assert r['condition_bank']==profile['conditions'] and r['normal_inference_only_EEG'] and not r['query_target_or_concept_network_input']
      assert r['UNet_NFE']==4 and r['EEG_deployment_forward_calls']==1 and p.is_relative_to(out/'images'/domain/arm)
     duplicates.setdefault(r['pixels_sha256'],[]).append([domain,arm,i])
     if np.max(rgb.astype(float).std((0,1)))<2.55:near.append([domain,arm,i])
     ordered.append(dict(r,domain=domain))
  assert len(ordered)==2400;K.write(out/'ordered_images.json',ordered)
  K.write(out/'image_audit.json',dict(status='PASS_ALL2400_IMAGE_PROVENANCE',new_images=1200,reused_images=1200,ordered_records=K.ref(out/'ordered_images.json'),all_paths_inodes_distinct=True,
   all_conditions_FP32_FP16_tokens_noises_pixels_verified=True,near_constant=near,duplicate_pixel_groups=[v for v in duplicates.values() if len(v)>1],raw_event_to_stimulus_replay_not_established=True))
 else:
  assert not (out/'readout.json').exists() and K.read(out/'image_audit.json')['status']=='PASS_ALL2400_IMAGE_PROVENANCE'
  import score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v2 as S
  import score_low_level as L
  import enigma_R80_count_image_contract_20261104_v1 as R
  assert K.ref(S.__file__)==q['runtime_sources']['R4_metrics'] and K.ref(L.__file__)==q['runtime_sources']['low_level']
  allarrays={};readout={};matrices={};diffs={}
  for domain in ['R4','R80']:
   n=400 if domain=='R4' else 200;enc=K.read(out/('encode_'+domain+'_completion.json'));assert enc['status']=='COMPLETE_FROZEN_DINO_FULL_DOMAIN'
   f=np.load(K.checked(enc['features']));target=np.load(K.checked(enc['target']));assert f.shape==(n,4,768) and target.shape==(n,768)
   rows=c['queries'][domain];arrays={}
   check=S.metric_arrays(target,target,rows,rows) if domain=='R4' else R.metrics(target,target)[0]
   assert np.all(check['DINO2AFC']==1) and np.all(check['gallery_top1']==1)
   for j,arm in enumerate(C.ARMS):
    if domain=='R4':
     arrays[arm]=S.metric_arrays(f[:,j],target,rows,rows)
     fn=f[:,j]/np.linalg.norm(f[:,j],axis=1,keepdims=True);tn=target/np.linalg.norm(target,axis=1,keepdims=True);matrix=fn@tn.T
    else:arrays[arm],matrix=R.metrics(f[:,j],target)
    matrices[domain+'__'+arm]=matrix
    for key in ['PixCorr_RGB','SSIM_Y','mean_RGB_absolute_error']:arrays[arm][key]=np.empty(n,np.float64)
    if domain=='R80':
     paper=K.read(out/('paper_'+arm+'_completion.json'));assert paper['status']=='COMPLETE_ALL200_AUTHOR_EIGHT_METRICS'
     per=np.load(K.checked(paper['values']));assert per.shape==(200,8)
     for k,key in enumerate(C.PAPER):arrays[arm][key]=per[:,k]
   metadata=K.read(K.checked(c['prepared']['target_metadata_ref']))['Dcheck'] if domain=='R4' else None;zips={}
   try:
    for i in range(n):
     guard()
     if domain=='R4':
      m=metadata[i]
      if m['image_zip'] not in zips:zips[m['image_zip']]=zipfile.ZipFile(m['image_zip'])
      raw=zips[m['image_zip']].read(m['image_member']);assert hashlib.sha256(raw).hexdigest()==m['image_sha256'];trgb,ty=L.pixels(io.BytesIO(raw))
     else:trgb,ty=L.pixels(K.checked(rows[i]['target']))
     tm=L.moments(ty)
     for arm in C.ARMS:
      rgb,y=L.pixels(K.checked(records[domain][arm][i]['image']));arrays[arm]['PixCorr_RGB'][i]=L.pixcorr(rgb,trgb);arrays[arm]['SSIM_Y'][i]=L.ssim_y(y,ty,tm)
      arrays[arm]['mean_RGB_absolute_error'][i]=float(abs(rgb.mean((0,1))-trgb.mean((0,1))).mean())
   finally:
    for z in zips.values():z.close()
   pairs,d=C.paired(arrays,rows,domain);diffs.update({domain+'__'+k:v for k,v in d.items()})
   allarrays.update({domain+'__'+arm+'__'+key:v for arm,rr in arrays.items() for key,v in rr.items()})
   readout[domain]=dict(queries=n,subjects=1,physical_R=4 if domain=='R4' else 80,gallery=n,metrics=list(C.R4_METRICS if domain=='R4' else C.R80_METRICS),
    means={arm:{k:float(v.mean()) if np.isfinite(v).all() else None for k,v in rr.items()} for arm,rr in arrays.items()},
    defined_counts={arm:{k:int(np.isfinite(v).sum()) for k,v in rr.items()} for arm,rr in arrays.items()},fixed_comparisons=pairs)
  for name,data in [('metrics.npz',allarrays),('cosine_matrices.npz',matrices),('paired_differences.npz',diffs)]:
   p=out/name;assert not p.exists()
   with p.open('xb') as f:np.savez_compressed(f,**data)
  K.write(out/'readout.json',dict(status='COMPLETE_CONTINUOUS_MARGIN_R4_R80_ALL_METRICS',domains=readout,arrays=K.ref(out/'metrics.npz'),matrices=K.ref(out/'cosine_matrices.npz'),paired_differences=K.ref(out/'paired_differences.npz'),
   all2400_images_before_effect_read=True,bootstrap=dict(seed=1729,replicates=1000,unit='concept_within_each_domain'),distance_metrics_lower_better=['mean_RGB_absolute_error','eval_effnet','eval_swav'],
   no_R80_within_concept10_defined=True,exposed_single_subject_development_NOT_independent_confirmation=True,all_tradeoffs_preserved=True,no_automatic_more_training=True,second_originality_or_SOTA_NOT_established=True))
 guard();C.resources(c);K.write(out/(a.mode+'_completion.json'),dict(status='COMPLETE_'+a.mode,seconds=time.monotonic()-start,extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0))
except BaseException as exc:
 K.write(out/(a.mode+'_failure.json'),dict(error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-start));raise
