"""Recompute all seven-method ranks, pixel metrics, differences and 1000 paired draws."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,io,json,time,traceback,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import brainhive_native_direction_image_contract_20261103_v1 as I
KEYS=('target_cosine','DINO2AFC','gallery_top1','within_concept10_top1','within_concept10_two_afc','PixCorr_RGB','SSIM_Y','mean_RGB_absolute_error')
def independent_ranks(features,gallery,queries,panel):
 z=features/np.linalg.norm(features,axis=1,keepdims=True);g=gallery/np.linalg.norm(gallery,axis=1,keepdims=True);sim=z@g.T;where={v['stimulus_id']:i for i,v in enumerate(panel)};positions=np.array([where[v['stimulus_id']] for v in queries]);out=np.empty((len(queries),5),np.float64)
 for i,(query,pos) in enumerate(zip(queries,positions)):
  right=sim[i,pos];subset=np.array([j for j,v in enumerate(panel) if v['concept_id']==query['concept_id']]);assert len(subset)==10
  out[i]=[right,(np.count_nonzero(sim[i]<right)+.5*(np.count_nonzero(sim[i]==right)-1))/399.,float(int(np.argmax(sim[i]))==pos),float(subset[int(np.argmax(sim[i,subset]))]==pos),
   (np.count_nonzero(sim[i,subset]<right)+.5*(np.count_nonzero(sim[i,subset]==right)-1))/9.]
 return out
def independent_low(path,trgb,ty):
 with Image.open(path) as im:rgb=im.convert('RGB').resize((256,256),Image.Resampling.BICUBIC);x=np.asarray(rgb,np.float64)/255.;y=np.asarray(rgb.convert('L'),np.float64)/255.
 xv=x.reshape(-1)-x.mean();tv=trgb.reshape(-1)-trgb.mean();den=np.sqrt(np.dot(xv,xv)*np.dot(tv,tv));corr=float(np.dot(xv,tv)/den) if den>0 else 0.
 filt=lambda v:gaussian_filter(v,sigma=1.5,truncate=3.5,mode='reflect');a=filt(y);b=filt(ty);va=filt(y*y)-a*a;vb=filt(ty*ty)-b*b;cov=filt(y*ty)-a*b
 ssim=((2*a*b+.0001)*(2*cov+.0009))/((a*a+b*b+.0001)*(va+vb+.0009));mae=np.abs(x.mean((0,1))-trgb.mean((0,1))).sum()/3.
 return np.array([corr,float(ssim[5:-5,5:-5].mean()),float(mae)])
def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--stage',type=int,choices=(80,400),required=True);a=p.parse_args();started=time.monotonic();s,train,e,old,base,G,H,C=I.setup(a.protocol);root=Path(s['output_root']);out=root/f'audit{a.stage}';assert (out/'inputs.json').exists() and not (out/'scores.json').exists();before=I.resources(s,train,e,C)
 try:
  assert I.dependency(s,train,C)['status']=='READY_ALL30_FIXED_FINALS_AND_ACTUAL_OWNER_EXIT';score=C.read(root/f'score{a.stage}/completion.json');assert score['status']=='COMPLETE_ALL7_METHODS_ALL8_FIXED_METRICS_NATIVE_EEG_POSTFIT_READOUT' and score['queries']==a.stage and score['protocol']==C.ref(a.protocol) and tuple(score['metrics'])==KEYS
  saved=dict(np.load(C.checked(score['raw']),allow_pickle=False));asset=I.assets(s,base,C,True);queries=asset['panel'][:a.stage];parts=[];rows=[]
  for lo in range(0,a.stage,80):
   done=C.read(root/'generation'/f'{lo:03d}_{lo+80:03d}'/'completion.json');parts.append(np.load(C.checked(done['evidence']['features']),allow_pickle=False));rows.extend(json.loads(v) for v in C.checked(done['evidence']['manifest']).read_text().splitlines())
  features=np.concatenate(parts);sources={arm:asset['old_features'][:a.stage,j] for j,arm in enumerate(I.OLD_ARMS)};sources.update({arm:features[:,j] for j,arm in enumerate(I.ARMS)});assert np.array_equal(saved['gallery_features'],asset['target_features'])
  values={};largest=0.
  for arm,f in sources.items():assert np.array_equal(f,saved['features__'+arm]);values[arm]=np.empty((a.stage,8));values[arm][:,:5]=independent_ranks(f,asset['target_features'],queries,asset['panel'])
  archives={}
  try:
   for i,q in enumerate(queries):
    meta=asset['metadata'][i]
    if meta['image_zip'] not in archives:archives[meta['image_zip']]=zipfile.ZipFile(meta['image_zip'])
    blob=archives[meta['image_zip']].read(meta['image_member']);assert hashlib.sha256(blob).hexdigest()==meta['image_sha256']
    with Image.open(io.BytesIO(blob)) as im:rgb=im.convert('RGB').resize((256,256),Image.Resampling.BICUBIC);trgb=np.asarray(rgb,np.float64)/255.;ty=np.asarray(rgb.convert('L'),np.float64)/255.
    for j,arm in enumerate(sources):
     row=asset['old_manifest'][3*i+j] if j<3 else rows[4*i+j-3];values[arm][i,5:]=independent_low(row['generated_path'],trgb,ty)
    if (i+1)%80==0:assert time.monotonic()-started<s['CPU_stage_seconds']-30
  finally:
   for archive in archives.values():archive.close()
  for arm,v in values.items():
   error=float(np.max(abs(v-saved['values__'+arm])));largest=max(largest,error);assert error<2e-6
   assert np.max(abs(v.mean(0)-np.array([score['means'][arm][key] for key in KEYS])))<2e-6
  concepts=sorted({v['concept_id'] for v in queries});groups=[[i for i,v in enumerate(queries) if v['concept_id']==concept] for concept in concepts];differences={};bootstrap_error=0.
  for name,(left,right) in I.CONTRASTS.items():
   delta=values[left]-values[right];differences[name]=delta;assert np.max(abs(delta-saved['difference__'+name]))<2e-6;pc=np.stack([delta[g].mean(0) for g in groups]);rng=np.random.default_rng(1729)
   samples=np.stack([pc[rng.integers(0,len(groups),size=len(groups))].mean(0) for _ in range(1000)]);ci=np.quantile(samples,[.025,.975],axis=0)
   for j,key in enumerate(KEYS):
    want=score['paired'][name][key];error=max(abs(float(delta[:,j].mean())-want['difference']),float(np.max(abs(ci[:,j]-want['descriptive_concept_bootstrap95']))));bootstrap_error=max(bootstrap_error,error);assert error<2e-6
  primary=score['paired']['directional_minus_old_ordinary'];afc=primary['DINO2AFC'];wc=primary['within_concept10_two_afc'];joint=afc['difference']<=-.02 and afc['descriptive_concept_bootstrap95'][1]<0 and wc['descriptive_concept_bootstrap95'][1]<0
  practical=afc['difference']>=.01 and afc['descriptive_concept_bootstrap95'][0]>0 and wc['difference']>=0;matched=all(score['paired'][name]['DINO2AFC']['descriptive_concept_bootstrap95'][0]>0 for name in ('directional_minus_positive_only','directional_minus_wrong_EEG','directional_minus_reencoded_ordinary'))
  expected=('STOP_EXPANSION_JOINT_HARM' if joint else 'EXPAND400') if a.stage==80 else ('PROMISING_MATCHED_INCREMENT_REQUIRES_MORE_SUBJECTS_AND_FULL_METRICS' if practical and matched else 'NO_CONFIRMED_MATCHED_INCREMENT_ON_THIS_EXPOSED_PANEL');assert score['decision']['action']==expected
  np.savez_compressed(out/'independent_metric_values_and_differences.npz',**{'values__'+arm:v for arm,v in values.items()},**{'difference__'+name:v for name,v in differences.items()})
  summary=dict(status='PASS_INDEPENDENT_ALL7_METHODS_ALL8_METRICS_RANKS_PIXELS_AND1000_PAIRED_DRAWS',UTC=G.datetime.datetime.now(G.datetime.timezone.utc).isoformat(),protocol=C.ref(a.protocol),source=C.ref(__file__),queries=a.stage,
   inputs_audit=C.ref(out/'inputs.json'),score_completion=C.ref(root/f'score{a.stage}/completion.json'),metric_cells=a.stage*7*8,difference_cells=a.stage*6*8,maximum_metric_abs_error=largest,maximum_bootstrap_abs_error=bootstrap_error,
   action_recomputed=expected,all_registered_denominators_retained=True,independent_values=C.ref(out/'independent_metric_values_and_differences.npz'),
   scope='All feature ranks and lowlevel values rederived from actual arrays and pixels; all1000 concept draws and fixed contrasts recomputed. Frozen DINO feature extraction itself has original-cache replay/provenance, not a second all-image DINO forward.',
   new_GPU_jobs=0,new_images=0,NEW_contacted=False,seconds=time.monotonic()-started,resources_before=before,resources_after=I.resources(s,train,e,C))
  assert time.monotonic()-started<s['CPU_stage_seconds'];G.write(out/'scores.json',summary);print(json.dumps(summary),flush=True)
 except BaseException as error:G.write(out/'scores_failure.json',dict(error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-started,no_automatic_retry=True));raise
if __name__=='__main__':main()
