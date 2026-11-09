"""Independent identities, full image metrics and fixed factorial contrasts."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import argparse,hashlib,io,json,resource,sys,time,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import atm_new_mean_factorial_contract_20261105_v2 as C
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--mode',choices=['prepare','audit','score'],required=True);a=ap.parse_args()
start=time.monotonic();rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
c=C.setup(a.policy);K,q,out=c['K'],c['q'],c['out'];C.resources(c);assert K.ref(__file__)==q['sources']['CPU']
def guard():
 assert time.monotonic()-start<850 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss<c['prior']['parent']['maximum_extra_CPU_bytes']
def save_npz(path,data):
 assert not path.exists()
 with path.open('xb') as f:np.savez_compressed(f,**data)
if a.mode=='prepare':
 assert not (out/'prepared.json').exists()
 C.save_array(out/'conditions.npy',c['values']);C.save_array(out/'native_tokens.npy',c['native']);C.save_array(out/'factorial_tokens.npy',c['bank']);C.save_array(out/'donors.npy',c['donors'])
 K.write(out/'queries.json',c['rows'])
 # Independently reconstruct additive decomposition in float64 before the actual FP16 cast.
 p=c['native'].astype(np.float64);m=p.mean(1,keepdims=True);center=p-m;d=c['donors']
 exact=np.stack([p,center+m[d],center[d]+m,p[d]],1)
 assert np.array_equal(exact.astype(np.float16),c['bank'])
 assert np.array_equal(C.tokens(c['native'],np.arange(400)),np.repeat(c['native'][:,None],4,axis=1))
 assert np.array_equal(np.sort(d),np.arange(400)) and all(c['rows'][i]['concept_id']==c['rows'][j]['concept_id'] and i!=j for i,j in enumerate(d))
 K.write(out/'prepared.json',dict(status='PASS_FULL400_FACTORIAL_INPUTS',policy=K.ref(a.policy),conditions=K.ref(out/'conditions.npy'),
  native=K.ref(out/'native_tokens.npy'),tokens=K.ref(out/'factorial_tokens.npy'),donors=K.ref(out/'donors.npy'),queries=K.ref(out/'queries.json'),
  all1600_FP16_tokens_independent_rebuild=True,zero_intervention_exact=True,within_concept_marginal_permutations=True,
  same_concept_labels_privileged_diagnostic=True,not_deployable=True,new_updates=0))
else:
 assert K.read(out/'prepared.json')['policy']==K.ref(a.policy)
 records=C.records(c)
 if a.mode=='audit':
  paths=set();inodes=set();duplicates={};near_constant=0
  for i in range(400):
   guard()
   for j,arm in enumerate(C.ARMS):
    r=records[arm][i];assert all(r[k]==c['rows'][i][k] for k in C.IDENTITY)
    p=K.checked(r['image']);s=p.stat();inode=(s.st_dev,s.st_ino)
    assert str(p) not in paths and inode not in inodes;paths.add(str(p));inodes.add(inode)
    px=np.asarray(Image.open(p).convert('RGB'));assert px.shape==(512,512,3) and C.sha(px)==r['pixels_sha256']
    assert r['noise']==records['correct'][i]['noise'] and len(r['noise']['initial'])==1 and len(r['noise']['ancestral'])==4
    assert r['condition_FP32_sha256']==C.sha(c['values'][i]) and r['condition_FP16_sha256']==C.sha(c['values'][i].astype(np.float16))
    assert r['tokens_FP16_sha256']==C.sha(c['bank'][i,j])
    if j:
     d=int(c['donors'][i]);assert r['donor_query_index']==d and r['donor_query_id']==c['rows'][d]['query_id']
     assert r['mean_donor_index']==(d if j in (1,3) else i) and r['center_donor_index']==(d if j in (2,3) else i)
     assert r['donor_tokens_FP16_sha256']==C.sha(c['native'][d]) and r['projection_calls']==r['UNet_NFE']==4
     assert r['privileged_same_concept_intervention'] and not r['deployable_method'] and not r['query_image_read_at_inference']
    duplicates.setdefault(r['pixels_sha256'],[]).append([arm,i]);near_constant+=int(px.astype(float).std((0,1)).max()<2.55)
  K.write(out/'ordered_images.json',[records[arm][i] for i in range(400) for arm in C.ARMS])
  K.write(out/'image_audit.json',dict(status='PASS_ALL1600_FACTORIAL_IMAGE_PROVENANCE',records=K.ref(out/'ordered_images.json'),
   images=1600,all_identity_condition_tokens_noise_pixels=True,near_constant_count=near_constant,duplicate_groups=[v for v in duplicates.values() if len(v)>1]))
 else:
  assert K.read(out/'image_audit.json')['status']=='PASS_ALL1600_FACTORIAL_IMAGE_PROVENANCE'
  import score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v2 as S
  import score_low_level as Low
  assert K.ref(S.__file__)==q['source_metrics'] and K.ref(Low.__file__)==q['source_low_level']
  f=np.load(K.checked(K.read(out/'encode_completion.json')['features']));target=np.load(K.checked(q['inputs']['target']))
  assert f.shape==(400,4,768) and target.shape==(400,768)
  sanity=S.metric_arrays(target,target,c['rows'],c['rows']);assert np.all(sanity['DINO2AFC']==1) and np.all(sanity['gallery_top1']==1)
  arrays={arm:S.metric_arrays(f[:,j],target,c['rows'],c['rows']) for j,arm in enumerate(C.ARMS)}
  prior_arrays=np.load(K.checked(q['inputs']['metrics']))
  prefix='R4__frozen_positive_NCE__'
  for key in C.METRICS[:5]:assert np.array_equal(arrays['correct'][key],prior_arrays[prefix+key]),key
  for key in C.METRICS[5:]:
   arrays['correct'][key]=prior_arrays[prefix+key].copy()
   for arm in C.ARMS[1:]:arrays[arm][key]=np.empty(400)
  metadata=K.read(K.checked(c['prior']['prepared']['target_metadata_ref']))['Dcheck'];archives={}
  try:
   for i,item in enumerate(metadata):
    guard();assert all(item[k]==c['rows'][i][k] for k in C.IDENTITY)
    if item['image_zip'] not in archives:archives[item['image_zip']]=zipfile.ZipFile(item['image_zip'])
    data=archives[item['image_zip']].read(item['image_member']);assert hashlib.sha256(data).hexdigest()==item['image_sha256']
    trgb,ty=Low.pixels(io.BytesIO(data));tm=Low.moments(ty)
    for arm in C.ARMS[1:]:
     rgb,y=Low.pixels(K.checked(records[arm][i]['image']));arrays[arm]['PixCorr_RGB'][i]=Low.pixcorr(rgb,trgb)
     arrays[arm]['SSIM_Y'][i]=Low.ssim_y(y,ty,tm);arrays[arm]['mean_RGB_absolute_error'][i]=float(abs(rgb.mean((0,1))-trgb.mean((0,1))).mean())
  finally:
   for z in archives.values():z.close()
  pairs,diffs=C.paired(arrays,c['rows']);donor_rows=[c['rows'][int(i)] for i in c['donors']]
  donor_arrays={arm:S.metric_arrays(f[:,j],target,donor_rows,c['rows']) for j,arm in enumerate(C.ARMS)}
  save_npz(out/'metrics.npz',{arm+'__'+key:v for arm,row in arrays.items() for key,v in row.items()})
  save_npz(out/'donor_metrics.npz',{arm+'__'+key:v for arm,row in donor_arrays.items() for key,v in row.items()})
  save_npz(out/'paired_differences.npz',diffs)
  fn=f/np.linalg.norm(f,axis=2,keepdims=True);tn=target/np.linalg.norm(target,axis=1,keepdims=True)
  C.save_array(out/'similarities.npy',np.stack([fn[:,j]@tn.T for j in range(4)],1))
  K.write(out/'readout.json',dict(status='COMPLETE_FULL400_NEW_MEAN_FACTORIAL_DIAGNOSTIC',arms=list(C.ARMS),queries=400,subjects=1,
   means={a:{k:float(v.mean()) for k,v in row.items()} for a,row in arrays.items()},contrasts=pairs,
   donor_means={a:{k:float(v.mean()) for k,v in row.items()} for a,row in donor_arrays.items()},
   metrics=K.ref(out/'metrics.npz'),donor_metrics=K.ref(out/'donor_metrics.npz'),paired_differences=K.ref(out/'paired_differences.npz'),similarities=K.ref(out/'similarities.npy'),
   all64_CIs_reported=True,not_independent_confirmation=True,not_deployable_method=True,novelty_or_SOTA=False,
   interpretation='Assess two conditional contributions and interaction jointly with stimulus/quality metrics; joint-distribution changes and FP16 rounding remain limitations. No winner or new fit selected by one threshold.'))
guard();C.resources(c);print(json.dumps(dict(status='COMPLETE_'+a.mode,seconds=time.monotonic()-start)),flush=True)
