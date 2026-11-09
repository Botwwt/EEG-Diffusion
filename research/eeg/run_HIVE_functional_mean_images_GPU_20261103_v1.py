"""Native four-step images with fixed response calibration and matched controls."""
import argparse,gc,hashlib,json,time,traceback
from pathlib import Path
import brainhive_functional_mean_contract_20261103_v1 as G

NOISE_KEYS=('initial_noise_latents_sha256','initial_UNet_input_sha256','ancestral_noise_observations')

def main():
 p=argparse.ArgumentParser();p.add_argument('--image-protocol',required=True);p.add_argument('--mode',choices=['profile','generate'],required=True);p.add_argument('--lo',type=int,default=0);p.add_argument('--hi',type=int,default=80);p.add_argument('--job-name',required=True);p.add_argument('--max-seconds',type=int,required=True);a=p.parse_args()
 started=time.monotonic();image_spec=G.read(a.image_protocol)
 assert image_spec['status']=='REGISTERED_FUNCTIONAL_MEAN_IMAGES_WITH_PAIRING_CONTROLS' and image_spec['seed']==1729
 for pin in image_spec['source_pins']:G.checked(pin)
 s=G.setup(G.checked(image_spec['base_protocol']));G.wrapper(s,a.job_name);before=G.resources(s,start=True);base=G.rss()
 import numpy as np,torch
 from PIL import Image
 from brainhive_component_generator_v1 import load_native
 from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
 from generate_BrainHIVE_neural_clean_fastscreen_20261027_v1 import sample_observed,save_image
 from brainhive_functional_mean_integration_20261103_v1 import FunctionalMeanField,ARMS,background
 root=Path(s['output_root']);stats=G.read(G.checked(image_spec['statistics_completion']));audit=G.read(G.checked(image_spec['statistics_CPU_audit']))
 assert stats['status']=='COMPLETE_ALL10_REAL_OOF_T_TOKEN_MEAN_STATISTICS_NO_SGD' and audit['status']=='PASS_ALL_T_SUFFICIENT_STATS_AND_FIXED16_INDEPENDENT_CPU64_AUDIT'
 assert stats['protocol']==image_spec['base_protocol'] and stats['fields']==audit['source_files']['fields']
 fields=dict(np.load(G.checked(stats['fields']),allow_pickle=False));assert list(ARMS)==image_spec['image_arms']
 d09=G.read(G.checked(s['future_image_refs']['D09_completion']));inputs=G.read(G.checked(d09['inputs']));panel=G.read(G.checked(inputs['panel']))['Dcheck']
 bank=np.load(G.checked(d09['check_conditions']),allow_pickle=False);rows=[json.loads(x) for x in G.checked(d09['check_manifest']).read_text().splitlines()]
 assert len(panel)==400 and bank.shape==(400,3,1024) and len(rows)==1200
 for i,query in enumerate(panel):
  for j in range(3):
   for k,v in query.items():assert rows[3*i+j][k]==v
 groups={}
 for i,query in enumerate(panel):groups.setdefault(query['concept_id'],[]).append(i)
 donor=list(range(400))
 for indices in groups.values():
  assert len(indices)==10
  for k,i in enumerate(indices):donor[i]=indices[(k+1)%10]
 assert all(donor[i]!=i and panel[donor[i]]['concept_id']==panel[i]['concept_id'] for i in range(400))
 if a.mode=='profile':out=root/'profile';assert a.max_seconds==420
 else:
  assert a.max_seconds==900 and a.lo in (0,80,160,240,320) and a.hi==a.lo+80
  profile=G.read(G.checked(image_spec['profile_completion'])) if 'profile_completion' in image_spec else G.read(root/'profile/completion.json')
  assert profile['status']=='PASS_NATIVE_PIXELS_ALL70_REAL_Q_AND_NEW_FIELD_ENGINEERING'
  out=root/'generation'/f'{a.lo:03d}_{a.hi:03d}'
 assert not out.exists();out.mkdir(parents=True);stage='LOAD_NATIVE';field=None
 def budget(refresh=False):G.budget(s,started,a.max_seconds,base,torch,refresh=refresh)
 def digest(module):
  h=hashlib.sha256()
  for key,value in module.state_dict().items():h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
  return h.hexdigest()
 try:
  torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  torch.cuda.set_per_process_memory_fraction(min(1.,(20<<30)/torch.cuda.get_device_properties(0).total_memory));torch.cuda.reset_peak_memory_stats()
  loader_dir=out/'loader';loader_dir.mkdir();observer=ConvertedWeightAudit(torch,loader_dir);observer.install()
  try:pipe,A,P,prompts,host=load_native(training=False)
  finally:observer.restore()
  loader=observer.complete();assert host==s['future_image_refs']['expected_host']
  frozen={k:digest(m) for k,m in dict(UNet=pipe.unet,VAE=pipe.vae,A=A,P=P).items()}
  field=FunctionalMeanField(pipe,P,fields,torch)
  np.savez(out/'actual_bias_fields.npz',**field.field_arrays())
  def render(i,arm,path,mode=None,capture=False,rounding=False):
   source_i=donor[i] if arm in ('ordinary_shared_donor','ordinary_donor') else i
   b=background(arm);raw=np.ascontiguousarray(bank[source_i,b:b+1]);field.reset(mode,capture_Q=capture,record_rounding=rounding)
   with torch.inference_mode():
    c=torch.from_numpy(raw.copy()).cuda();tokens=P(A(c.half()));image,observed=sample_observed(pipe,prompts,tokens,panel[i]['image_seed'],torch)
   calls=field.proof()
   for key in NOISE_KEYS:assert observed[key]==rows[3*i][key]
   saved=save_image(path,image)
   record=panel[i]|dict(arm=arm,query_index=i,EEG_source_query_id=panel[source_i]['query_id'],EEG_source_stimulus_id=panel[source_i]['stimulus_id'],EEG_source_array_row=panel[source_i]['array_row'],
    intervention='same_concept_EEG_cycle' if source_i!=i else 'correct',condition_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),condition_FP16_sha256=hashlib.sha256(raw.astype(np.float16).tobytes()).hexdigest(),tokens_sha256=hashlib.sha256(tokens.cpu().contiguous().numpy().tobytes()).hexdigest(),
    ordinary=s['ordinary'] if b==1 else None,fields=stats['fields'],attention_call_proof=calls,normal_inference_target_access=False,one_image_per_query=True,UNet_policy_NFE=4)|saved|observed
   return record,tokens.detach().cpu().double().numpy()[0]
  if a.mode=='profile':
   stage='ZERO_NATIVE_PIXEL_PARITIES';parities=[]
   for arm,mode,reference_index in (('native_shared_mean',None,0),('ordinary_shared_mean',None,1),('ordinary_shared_mean','zero',1)):
    capture=reference_index==1 and mode is None
    row,tokens=render(0,arm,out/f'baseline_{reference_index}_{mode}.webp',mode=mode,capture=capture)
    old=rows[reference_index]
    assert row['condition_sha256']==old['condition_sha256'] and row['condition_FP16_sha256']==old['condition_FP16_sha256'] and row['tokens_sha256']==old['tokens_sha256']
    assert row['generated_pixel_sha256']==old['generated_pixel_sha256'] and row['generated_sha256']==old['generated_sha256']
    parities.append(row)
    if capture:
     with torch.inference_mode():other=P(A(torch.from_numpy(np.ascontiguousarray(bank[donor[0],1:2])).cuda().half())).cpu().double().numpy()[0]
     real_Q=field.real_Q_algebra(tokens,other)
     np.savez_compressed(out/'actual_Q_slices.npz',**{name:np.stack(values) for name,values in field.queries.items()})
   stage='ALL_FIXED_ENGINEERING_FIELDS';changed=[]
   for arm in ARMS:
    budget(refresh=True);row,_=render(0,arm,out/(arm+'.webp'),mode=arm,rounding=True);changed.append(row)
   numeric=dict(real_Q=real_Q,maximum_FP16_hook_rounding_error=field.maximum_rounding_error,maximum_registered_rounding_bound=field.maximum_rounding_bound,
    wrong_field_per_layer_norms_preserved=True,all_original_keys_values_processors_unmodified_for_shared_fields=True)
   G.write(out/'engineering_rows.json',dict(parities=parities,changed=changed,numerics=numeric))
   status='PASS_NATIVE_PIXELS_ALL70_REAL_Q_AND_NEW_FIELD_ENGINEERING';research_images=0
  else:
   stage='FIXED_IMAGES_NO_TARGET_INPUT';records=[];manifest=out/'manifest.jsonl'
   with manifest.open('x') as stream:
    for i in range(a.lo,a.hi):
     budget(refresh=(i%10==0))
     for arm in ARMS:
      row,_=render(i,arm,out/'pixels'/f'{i:03d}_{arm}.webp',mode=arm)
      stream.write(json.dumps(row)+'\n');stream.flush();records.append(row)
     if (i-a.lo+1)%10==0:print(json.dumps(dict(stage=stage,queries=i-a.lo+1,saved=len(records),seconds=time.monotonic()-started)),flush=True)
   assert len(records)==80*len(ARMS)
   status='COMPLETE_FIXED80_QUERY_SIX_ARM_IMAGE_SEGMENT';research_images=len(records)
  assert {k:digest(m) for k,m in dict(UNet=pipe.unet,VAE=pipe.vae,A=A,P=P).items()}==frozen
  field.close();field=None
  if a.mode=='generate':
   stage='FROZEN_DINO_NEW_IMAGES_ONLY';del pipe,A,P,prompts;gc.collect();torch.cuda.empty_cache()
   import image_evaluation as E
   model=E.dino_model();parts=[]
   for lo in range(0,len(records),32):
    budget();parts.append(E.dino_features([Path(row['generated_path']) for row in records[lo:lo+32]],model,batch_size=32,real_images=False).float().numpy())
   features=np.concatenate(parts).reshape(80,len(ARMS),768);assert np.isfinite(features).all()
   np.save(out/'generated_DINO_features.npy',features,allow_pickle=False)
   evidence=dict(manifest=G.ref(manifest),generated_features=G.ref(out/'generated_DINO_features.npy'),DINO_weight_SHA256=E.DINO_SHA256,
                 condition_source='Original fixedD09caches andsame frozenordinary; no targetfeatures enter sampler.',scores_computed=False)
  else:evidence=dict(engineering=G.ref(out/'engineering_rows.json'),numerics=numeric,actual_Q=G.ref(out/'actual_Q_slices.npz'),zero_native_and_ordinary_pixel_parities=3)
  budget(refresh=True)
  done=dict(status=status,UTC=G.datetime.datetime.now(G.datetime.timezone.utc).isoformat(),image_protocol=G.ref(a.image_protocol),statistics=image_spec['statistics_completion'],
   mode=a.mode,range=[a.lo,a.hi] if a.mode=='generate' else None,arms=list(ARMS),evidence=evidence,fields=stats['fields'],actual_bias_fields=G.ref(out/'actual_bias_fields.npz'),
   research_images=research_images,engineering_images=9 if a.mode=='profile' else 0,native_weights_unchanged=True,new_training_updates=0,one_image_per_query=True,new_UNet_NFE_per_query=4,
   loader=loader,host=host,normal_inference_target_access=False,ordinary_unchanged=True,NEW_contacted=False,seconds=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=G.rss()-base)
  G.write(out/'completion.json',done);print(json.dumps(done),flush=True)
 except BaseException as error:
  G.write(out/'failure.json',dict(stage=stage,error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-started,no_automatic_retry=True));raise
 finally:
  if field is not None:field.close()
if __name__=='__main__':main()
