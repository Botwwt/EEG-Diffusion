"""Fixed native four-step images; full final-model dependency and matched reexport baseline."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,gc,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import brainhive_native_direction_image_contract_20261103_v1 as I
NOISE_KEYS=('initial_noise_latents_sha256','initial_UNet_input_sha256','ancestral_noise_observations')
def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--mode',choices=('profile','generate'),required=True);p.add_argument('--lo',type=int,default=0);p.add_argument('--job-name',required=True);a=p.parse_args();started=time.monotonic()
 s,train,e,old,base,G,H,C=I.setup(a.protocol);assert I.dependency(s,train,C)['status']=='READY_ALL30_FIXED_FINALS_AND_ACTUAL_OWNER_EXIT';G.wrapper(base,a.job_name)
 before=I.resources(s,train,e,C,True,True);rss=G.rss();root=Path(s['output_root']);spec=C.read(root/'conditions/completion.json');assert spec['status']=='COMPLETE_ALL400_FOUR_ARM_FIXED_FINAL_CPU_EEG_CONDITIONS' and spec['protocol']==C.ref(a.protocol)
 assets=I.assets(s,base,C);panel=assets['panel'];rows=assets['old_manifest'];bank=np.load(C.checked(spec['conditions']),allow_pickle=False);assert bank.shape==(400,4,1024) and np.isfinite(bank).all()
 if a.mode=='profile':out=root/'profile';limit=s['profile_seconds']
 else:
  assert a.lo in (0,80,160,240,320);out=root/'generation'/f'{a.lo:03d}_{a.lo+80:03d}';limit=s['generation_segment_seconds'];profile=C.read(root/'profile/completion.json');assert profile['status']=='PASS_NATIVE_PARITIES_FOUR_FINAL_CONDITIONS_AND_DINO_CACHE_REPLAY'
 assert not out.exists();out.mkdir(parents=True);stage='LOAD';torch=None
 try:
  import torch
  from brainhive_component_generator_v1 import load_native
  from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
  from brainhive_native_direction_contract_20261103_v1 import tensor_digest
  from generate_BrainHIVE_neural_clean_fastscreen_20261027_v1 import sample_observed,save_image
  torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  torch.cuda.set_per_process_memory_fraction((20<<30)/torch.cuda.get_device_properties(0).total_memory);torch.cuda.reset_peak_memory_stats()
  def budget(refresh=False):
   assert time.monotonic()-started<limit-30 and G.rss()-rss<10<<30 and torch.cuda.max_memory_allocated()<20<<30
   if refresh:I.resources(s,train,e,C)
  loader_dir=out/'loader';loader_dir.mkdir();loader=ConvertedWeightAudit(torch,loader_dir);loader.install()
  try:pipe,A,P,prompts,host=load_native(training=False)
  finally:loader.restore()
  loading=loader.complete();assert host==assets['inputs']['expected_host'];modules=dict(UNet=pipe.unet,VAE=pipe.vae,A=A,P=P);frozen={k:tensor_digest(v) for k,v in modules.items()}
  def render(raw,query,path,arm,index,reference=None):
   budget();raw=np.ascontiguousarray(raw);assert raw.shape==(1,1024) and raw.dtype==np.float32
   with torch.inference_mode():tokens=P(A(torch.from_numpy(raw.copy()).cuda().half()));image,observed=sample_observed(pipe,prompts,tokens,query['image_seed'],torch)
   proof=dict(condition_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),condition_FP16_sha256=hashlib.sha256(raw.astype(np.float16).tobytes()).hexdigest(),tokens_sha256=hashlib.sha256(tokens.cpu().contiguous().numpy().tobytes()).hexdigest())
   if reference is not None:
    for key in NOISE_KEYS:assert observed[key]==reference[key]
   return query|observed|proof|save_image(path,image)|dict(arm=arm,query_index=index,condition_source=spec['conditions'],encoder_model=next(v['model'] for v in spec['models'] if v['arm']==arm) if arm in I.ARMS else None,
    ordinary=base['ordinary'] if arm in I.ARMS or arm=='ordinary_Dcheck' else None,evaluation_target_SHA256=assets['metadata'][index]['image_sha256'] if index>=0 else None,
    EEG_source_query_id=query['query_id'],EEG_source_array_row=query.get('array_row'),normal_inference_target_access=False,
    normal_inference_only_current_query_EEG=True,one_image_per_query=True,UNet_policy_NFE=4)
  stage='EXACT_CACHED_NATIVE_T_AND_ORDINARY_PARITIES';parities=[];records=[];manifest=None
  if a.mode=='profile':
   cache=np.load(C.checked(assets['D09']['check_conditions']),allow_pickle=False)
   for label,raw,reference in [('native_Dcheck',cache[0,0:1],rows[0]),('ordinary_Dcheck',cache[0,1:2],rows[1]),('native_T',np.load(C.checked(assets['inputs']['parity_condition']),allow_pickle=False),assets['inputs']['parity_reference'])]:
    query=dict(image_seed=reference['image_seed'],query_id=reference['query_id']);record=render(np.array(raw,copy=True),query,out/(label+'.webp'),label,-1)
    for key in ('initial_noise_latents_sha256','initial_UNet_input_sha256','condition_sha256','condition_FP16_sha256','tokens_sha256','generated_sha256','generated_pixel_sha256'):assert record[key]==reference[key]
    if 'ancestral_noise_observations' in reference:assert record['ancestral_noise_observations']==reference['ancestral_noise_observations']
    parities.append(record)
   stage='FIXED_FOUR_FINAL_ENGINEERING_ROWS_REUSED_BY_STAGE80'
   for j,arm in enumerate(I.ARMS):records.append(render(bank[0,j:j+1],panel[0],out/(arm+'.webp'),arm,0,rows[0]))
   G.write(out/'engineering_rows.json',dict(parities=parities,reusable_query0_rows=records));evidence=dict(engineering=C.ref(out/'engineering_rows.json'),parities=3,reusable_research_images=4)
  else:
   stage='FIXED_EEG_ONLY_IMAGE_SEGMENT';manifest=out/'manifest.jsonl'
   if a.lo==0:
    engineering=C.read(C.checked(profile['evidence']['engineering']));records=list(engineering['reusable_query0_rows']);assert len(records)==4
    for j,row in enumerate(records):assert row['query_index']==0 and row['arm']==I.ARMS[j] and C.sha(row['generated_path'])==row['generated_sha256']
   with manifest.open('x') as stream:
    for row in records:stream.write(json.dumps(row)+'\n')
    for i in range(a.lo+(1 if a.lo==0 else 0),a.lo+80):
     budget(i%10==0)
     for j,arm in enumerate(I.ARMS):
      record=render(bank[i,j:j+1],panel[i],out/'pixels'/f'{i:03d}_{arm}.webp',arm,i,rows[3*i]);records.append(record);stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
     if (i+1)%10==0:print(json.dumps(dict(stage=stage,lo=a.lo,queries=i-a.lo+1,rows=len(records),seconds=time.monotonic()-started)),flush=True)
   assert len(records)==320
   evidence=dict(manifest=C.ref(manifest),reused_query0_images=4 if a.lo==0 else 0,new_images=316 if a.lo==0 else 320)
  assert {k:tensor_digest(v) for k,v in modules.items()}==frozen
  del modules,pipe,A,P,prompts;gc.collect();torch.cuda.empty_cache();stage='FROZEN_DINO_FEATURES'
  import image_evaluation as D
  assert C.ref(D.DINO_WEIGHT)==s['DINO_weight'];model=D.dino_model()
  if a.mode=='profile':
   actual=D.dino_features([Path(r['generated_path']) for r in rows[:3]],model,batch_size=32,real_images=False).float().numpy();expected=np.load(C.checked(assets['D09']['check_generated_features']),allow_pickle=False)[0]
   maxabs=float(np.max(abs(actual-expected)));cos=np.sum(actual*expected,axis=1)/(np.linalg.norm(actual,axis=1)*np.linalg.norm(expected,axis=1));assert np.allclose(actual,expected,atol=5e-5,rtol=1e-4) and cos.min()>=1-1e-6
   np.savez_compressed(out/'DINO_baseline_replay.npz',actual=actual,expected=expected);evidence['DINO_replay']=dict(arrays=C.ref(out/'DINO_baseline_replay.npz'),max_abs=maxabs,minimum_cosine=float(cos.min()),fixed_tolerances=dict(atol=5e-5,rtol=1e-4,min_cosine=1-1e-6))
   status='PASS_NATIVE_PARITIES_FOUR_FINAL_CONDITIONS_AND_DINO_CACHE_REPLAY'
  else:
   parts=[]
   for lo in range(0,len(records),32):budget();parts.append(D.dino_features([Path(r['generated_path']) for r in records[lo:lo+32]],model,batch_size=32,real_images=False).float().numpy())
   features=np.concatenate(parts).reshape(80,4,768);assert np.isfinite(features).all();np.save(out/'features.npy',features,allow_pickle=False);evidence['features']=C.ref(out/'features.npy');evidence['DINO_weight']=s['DINO_weight'];status='COMPLETE_FIXED80_FOUR_ARM_NATIVE_FINAL_IMAGE_SEGMENT'
  budget(True);complete=dict(status=status,UTC=G.datetime.datetime.now(G.datetime.timezone.utc).isoformat(),protocol=C.ref(a.protocol),source=C.ref(__file__),conditions=C.ref(root/'conditions/completion.json'),arms=list(I.ARMS),range=[a.lo,a.lo+80] if a.mode=='generate' else None,
   evidence=evidence,host=host,loader=loading,old_ordinary_immutable=True,all_native_host_weights_frozen=True,normal_inference_NFE=4,new_training_updates=0,normal_inference_target_access=False,NEW_contacted=False,seconds=time.monotonic()-started,
   peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=G.rss()-rss,resources_before=before,resources_after=I.resources(s,train,e,C))
  G.write(out/'completion.json',complete);print(json.dumps(complete),flush=True)
 except BaseException as error:G.write(out/'failure.json',dict(error=repr(error),stage=stage,traceback=traceback.format_exc(),seconds=time.monotonic()-started,no_automatic_retry=True));raise
if __name__=='__main__':main()
