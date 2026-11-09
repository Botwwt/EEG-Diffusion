"""Fixed seven-method DINO/detail readout after exhaustive image/input identity audit."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,csv,hashlib,io,json,time,traceback,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import brainhive_native_direction_image_contract_20261103_v1 as I
import brainhive_native_direction_readout_20261103_v1 as V
def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--stage',type=int,choices=(80,400),required=True);a=p.parse_args();started=time.monotonic()
 s,train,e,old,base,G,H,C=I.setup(a.protocol);assert I.dependency(s,train,C)['status']=='READY_ALL30_FIXED_FINALS_AND_ACTUAL_OWNER_EXIT';before=I.resources(s,train,e,C);root=Path(s['output_root']);out=root/f'score{a.stage}';assert not out.exists();out.mkdir()
 try:
  audit=C.read(root/f'audit{a.stage}/inputs.json');assert audit['status']=='PASS_ALL_FIXED_NATIVE_EEG_IMAGE_IDENTITIES_CONDITIONS_NOISE_PIXELS_BEFORE_SCORE' and audit['protocol']==C.ref(a.protocol) and audit['queries']==a.stage
  if a.stage==400:assert C.read(root/'score80/completion.json')['decision']['action']=='EXPAND400'
  import score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v2 as S
  import score_low_level as L
  assets=I.assets(s,base,C,True);queries=assets['panel'][:a.stage];oldrows=assets['old_manifest'];parts=[];newrows=[];segments=[]
  for lo in range(0,a.stage,80):
   path=root/'generation'/f'{lo:03d}_{lo+80:03d}'/'completion.json';done=C.read(path);assert done['status']=='COMPLETE_FIXED80_FOUR_ARM_NATIVE_FINAL_IMAGE_SEGMENT' and done['range']==[lo,lo+80] and done['protocol']==C.ref(a.protocol)
   parts.append(np.load(C.checked(done['evidence']['features']),allow_pickle=False));newrows.extend(json.loads(v) for v in C.checked(done['evidence']['manifest']).read_text().splitlines());segments.append(C.ref(path))
  features=np.concatenate(parts);assert features.shape==(a.stage,4,768) and len(newrows)==a.stage*4
  vectors={arm:assets['old_features'][:a.stage,j] for j,arm in enumerate(I.OLD_ARMS)};vectors.update({arm:features[:,j] for j,arm in enumerate(I.ARMS)})
  values={arm:np.empty((a.stage,8),np.float64) for arm in vectors}
  for arm,f in vectors.items():
   d=S.metric_arrays(f,assets['target_features'],queries,assets['panel'])
   for j,key in enumerate(V.DINO_METRICS):values[arm][:,j]=d[key]
  test=S.metric_arrays(assets['target_features'],assets['target_features'],assets['panel'],assets['panel']);assert np.all(test['DINO2AFC']==1) and np.all(test['gallery_top1']==1)
  checks=L.numerical_checks();archives={};target_shas=[]
  try:
   for i,q in enumerate(queries):
    meta=assets['metadata'][i]
    if meta['image_zip'] not in archives:archives[meta['image_zip']]=zipfile.ZipFile(meta['image_zip'])
    data=archives[meta['image_zip']].read(meta['image_member']);assert len(data)==int(meta['image_bytes']) and hashlib.sha256(data).hexdigest()==meta['image_sha256'];target_shas.append(meta['image_sha256'])
    with Image.open(io.BytesIO(data)) as opened:
     rgb=opened.convert('RGB').resize((256,256),Image.Resampling.BICUBIC);trgb=np.asarray(rgb,np.float64)/255.;ty=np.asarray(rgb.convert('L'),np.float64)/255.
    tm=L.moments(ty)
    for j,arm in enumerate(vectors):
     row=(oldrows[3*i+j] if j<3 else newrows[4*i+j-3]);assert row['query_id']==q['query_id'] and row['arm']==arm and C.sha(row['generated_path'])==row['generated_sha256']
     grgb,gy=L.pixels(row['generated_path']);values[arm][i,5:]=[L.pixcorr(grgb,trgb),L.ssim_y(gy,ty,tm),float(np.abs(grgb.mean((0,1))-trgb.mean((0,1))).mean())]
    if (i+1)%80==0:assert time.monotonic()-started<s['CPU_stage_seconds']-30;I.resources(s,train,e,C)
  finally:
   for archive in archives.values():archive.close()
  result,differences=V.summarize(values,queries);arrays={'gallery_features':assets['target_features']}
  arrays.update({'features__'+arm:f for arm,f in vectors.items()});arrays.update({'values__'+arm:v for arm,v in values.items()});arrays.update({'difference__'+name:v for name,v in differences.items()})
  np.savez_compressed(out/'all_features_metrics_differences.npz',**arrays)
  with (out/'per_query.csv').open('x',newline='') as stream:
   writer=csv.DictWriter(stream,fieldnames=['query_id','subject_id','concept_id','stimulus_id','array_row','image_seed','arm','target_image_SHA256',*V.METRICS]);writer.writeheader()
   for i,q in enumerate(queries):
    for arm,v in values.items():writer.writerow({k:q[k] for k in ('query_id','subject_id','concept_id','stimulus_id','array_row','image_seed')}|dict(arm=arm,target_image_SHA256=target_shas[i])|{key:float(v[i,j]) for j,key in enumerate(V.METRICS)})
  complete=dict(status='COMPLETE_ALL7_METHODS_ALL8_FIXED_METRICS_NATIVE_EEG_POSTFIT_READOUT',UTC=G.datetime.datetime.now(G.datetime.timezone.utc).isoformat(),protocol=C.ref(a.protocol),source=C.ref(__file__),queries=a.stage,concepts=a.stage//10,
   methods=list(vectors),metrics=list(V.METRICS),metric_cells=a.stage*7*8,difference_cells=a.stage*6*8,gallery_targets=400,segments=segments,pre_score_inputs_audit=C.ref(root/f'audit{a.stage}/inputs.json'),
   **result,raw=C.ref(out/'all_features_metrics_differences.npz'),per_query=C.ref(out/'per_query.csv'),all_target_images_SHA_verified=a.stage,lowlevel_pure_function_checks=checks,
   mean_RGB_absolute_error_lower_is_better=True,normal_inference_target_access=False,no_best_of_N=True,new_GPU_jobs=0,new_images=0,NEW_contacted=False,
   panel_already_exposed=True,single_subject_R4=True,single_training_seed=1729,independent_confirmation=False,publication_ready=False,SOTA=False,seconds=time.monotonic()-started,
   resources_before=before,resources_after=I.resources(s,train,e,C))
  assert time.monotonic()-started<s['CPU_stage_seconds'];G.write(out/'completion.json',complete);print(json.dumps(complete),flush=True)
 except BaseException as error:G.write(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-started,all_registered_denominators_retained=True,no_automatic_retry=True));raise
if __name__=='__main__':main()
