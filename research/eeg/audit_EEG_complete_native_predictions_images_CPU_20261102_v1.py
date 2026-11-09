"""Full final EEG forward and image identity/latent-law checks, without CUDA.

Captured private uniforms permit independent CPU inversion of the latent law.
They do not constitute CPU replay of native image generation or token RNG.
"""
import argparse,hashlib,os,time
from pathlib import Path
import numpy as np
import EEG_complete_native_study_contract_CPU_20261102_v1 as C

def seed(stimulus,kind):
 text='1729|EEG-native-marginal-v2|'+kind+'|sub-01|'+str(stimulus)+'|R4'
 return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'little')%(2**63-1)

def split_arm(arm):
 for suffix in ('_same','_different'):
  if arm.endswith(suffix):return arm[:-len(suffix)],suffix[1:]
 return arm,'correct'

def all_image_arms():return list(C.ARMS)+[a+s for a in C.CANDIDATES for s in ('_same','_different')]

def donor_maps(panel):
 groups={}
 for i,row in enumerate(panel):groups.setdefault(row['concept_id'],[]).append(i)
 assert len(panel)==400 and len(groups)==40 and all(len(v)==10 for v in groups.values())
 keys=sorted(groups)
 same={i:groups[k][(j+1)%10] for k in keys for j,i in enumerate(groups[k])}
 different={i:groups[keys[(n+1)%40]][j] for n,k in enumerate(keys) for j,i in enumerate(groups[k])}
 assert sorted(same.values())==sorted(different.values())==list(range(400))
 assert all(i!=same[i] and panel[i]['concept_id']==panel[same[i]]['concept_id'] and panel[i]['concept_id']!=panel[different[i]]['concept_id'] for i in range(400))
 return same,different

def independent_chain(logits,stay,arm,uniforms):
 """Sequential float64 inverse CDF; no production sampler import."""
 x=np.asarray(logits,np.float64);z=np.asarray(stay,np.float64)
 assert x.shape==(10,1000) and z.shape==(9,) and np.isfinite(x).all() and np.isfinite(z).all()
 p=np.exp(x-x.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
 base,_=split_arm(arm)
 if base=='chain_full':
  rho=np.empty_like(z);positive=z>=0;rho[positive]=1/(1+np.exp(-z[positive]));v=np.exp(z[~positive]);rho[~positive]=v/(1+v)
 elif base=='independent_full':rho=np.zeros(9)
 else:assert base in C.ARMS;rho=np.ones(9)
 first=np.asarray(uniforms['initial_uniform'],np.float64);keep=np.asarray(uniforms['stay_uniform'],np.float64);refresh=np.asarray(uniforms['refresh_uniform'],np.float64)
 assert first.shape==(1,1) and keep.shape==refresh.shape==(1,9)
 assert all(np.isfinite(v).all() and ((v>=0)&(v<1)).all() for v in (first,keep,refresh))
 cdf=np.cumsum(p,axis=-1)
 def draw(stage,u):return min(int(np.count_nonzero(cdf[stage]<u)),999)
 path=[draw(0,first[0,0])]
 for j in range(9):path.append(path[-1] if keep[0,j]<rho[j] else draw(j+1,refresh[0,j]))
 margin=min(float(np.min(np.abs(cdf[0]-first[0,0]))),*(float(np.min(np.abs(cdf[j+1]-refresh[0,j]))) for j in range(9)))
 return path,margin

def finals(protocol):
 result={}
 for arm in C.ARMS:
  root=C.OUT/'fits'/arm;done=C.read(root/'completion.json');cpu=C.read(root/'fit10_CPU_audit.json')
  assert done['status']=='COMPLETE_FIXED10_FULL_T_EEG_ARM' and cpu['status']=='PASS_ALL_FIXED_EEG_EPOCH_STATE_ADAM_COVERAGE'
  assert done['arm']==cpu['arm']==arm and done['epoch']==cpu['epoch']==10 and done['updates']==cpu['updates']==2070
  assert done['protocol']==cpu['protocol']==C.ref(protocol) and done['final_model']==cpu['final_model']
  assert Path(done['final_model']['path'])==root/'model_final.pt';C.checked(done['final_model']);result[arm]=done
 return result

def guard_factory(q,seconds):
 import resource
 start=time.monotonic();rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
 def guard():
  assert time.monotonic()-start<seconds and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss<q['maximum_extra_CPU_bytes']
  C.resources(q)
 return guard

def predictions(protocol):
 q=C.setup(protocol,formal=True);guard=guard_factory(q,q['CPU_prediction_seconds']);C.roots()
 import torch
 import EEG_complete_native_runtime_20261102_v1 as R
 torch.set_num_threads(2);assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 source=C.OUT/'tasks/predict_final400/completion.json';done=C.read(source);models=finals(protocol)
 assert done['status']=='COMPLETE_ALL6_FINAL_EEG_ONLY_DCHECK400_PREDICTIONS' and done['protocol']==C.ref(protocol)
 assert set(done['files'])==set(done['models'])==set(C.ARMS) and done['query_images_or_labels_read_by_predictor'] is False
 x=R.inputs(q,'Dcheck');channels=R.channels(q);errors={};cells=0
 tol=q['CPU_prediction_tolerance'];assert tol==dict(atol=5e-4,rtol=1e-4)
 for arm in C.ARMS:
  guard();assert done['models'][arm]==models[arm]['final_model']
  p=R.model(q,arm,'cpu');state=torch.load(C.checked(done['models'][arm]),map_location='cpu',weights_only=True)
  assert state['purpose']=='FORMAL_FIXED_FULL_T_FINAL_ONLY' and state['protocol']==C.ref(protocol) and state['arm']==arm and state['updates']==2070
  p.load_state_dict(state['model'],strict=True)
  with np.load(C.checked(done['files'][arm]),allow_pickle=False) as f:
   saved=f['logits'].copy();stay=f['stay'].copy()
  stages=10 if arm in ('chain_full','independent_full') else 1
  assert saved.shape==(400,stages,1000) and stay.shape==(400,9) and saved.dtype==stay.dtype==np.float32
  logits=[];rho=[]
  with torch.inference_mode():
   for lo in range(0,400,32):
    guard();v,z=p(torch.from_numpy(np.array(x[lo:lo+32])),input_chans=channels)
    logits.append((v[:,0:1] if stages==1 else v).numpy().copy());rho.append(z.numpy().copy())
   if arm=='image_prior_full':
    alternative=torch.from_numpy(np.array(x[::-1][:17])*7.0)
    alt,altz=p(alternative,input_chans=channels)
    assert torch.equal(alt,torch.from_numpy(saved[:17]).expand(-1,10,-1)) and torch.equal(altz,torch.from_numpy(stay[:17]))
    assert np.array_equal(saved,np.repeat(saved[:1],400,axis=0)) and np.array_equal(stay,np.zeros_like(stay))
  actual=np.concatenate(logits);z=np.concatenate(rho)
  assert np.isfinite(saved).all() and np.isfinite(stay).all() and np.isfinite(actual).all() and np.isfinite(z).all()
  assert np.allclose(saved,actual,**tol) and np.allclose(stay,z,**tol)
  errors[arm]=dict(all400_CPU_forward=True,logit_cells=int(actual.size),stay_cells=int(z.size),max_logits_error=float(np.abs(saved-actual).max()),max_stay_error=float(np.abs(stay-z).max()),final_model=done['models'][arm],prediction_file=done['files'][arm]);cells+=actual.size+z.size
  del p,state,actual,z
 assert not torch.cuda.is_initialized()
 C.write(C.OUT/'final_predictions_CPU_audit.json',dict(status='PASS_ALL6_FINAL400_CPU_FORWARD_AND_NOEEG_INDEPENDENCE',protocol=C.ref(protocol),source=C.ref(source),input=done['EEG_input'],tolerance=tol,errors=errors,all_prediction_cells=int(cells),noEEG_reversed_and_scaled_input_invariance=True,CUDA_initialized=False,normal_inference_target_access=False))

def artifacts(protocol,mode):
 q=C.setup(protocol,formal=mode!='profile');guard=guard_factory(q,q['CPU_artifact_seconds'])
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 from PIL import Image
 inputs=C.read(C.OUT/'inputs/completion.json');joined=C.read(C.checked(inputs['identities']));panel=C.read(C.checked(C.read(C.checked(q['legacy_image_spec']))['panel']))['queries']
 profile=mode=='profile';conditions={};model_sources={}
 if profile:
  audit=C.read(C.OUT/'all6_profiles_CPU_audit.json');assert audit['status']=='PASS_ALL6_REAL_PROFILE_GRADIENTS_AND_SERIALIZED_RESUME_STATES' and audit['protocol']==C.ref(protocol)
  for arm in C.ARMS:
   done=C.read(C.OUT/'tasks'/(arm+'_profile')/'completion.json')
   with np.load(C.checked(done['traces'][2]),allow_pickle=False) as f:
    assert f['rows'][0]==0;conditions[arm]=(f['logits'][:1].copy(),f['stay'][:1].copy())
   model_sources[arm]=dict(engineering_trace=done['traces'][2],purpose='BEFORE_TEMPORARY_UPDATE3_NOT_A_METHOD_MODEL')
  chunks=[C.OUT/'tasks/generation_profile'];arms=list(C.ARMS);rows=joined['T_rows'];expected=[0]
 else:
  models=finals(protocol);pred=C.read(C.OUT/'tasks/predict_final400/completion.json');cpu=C.read(C.OUT/'final_predictions_CPU_audit.json')
  assert cpu['status']=='PASS_ALL6_FINAL400_CPU_FORWARD_AND_NOEEG_INDEPENDENCE' and cpu['source']==C.ref(C.OUT/'tasks/predict_final400/completion.json') and cpu['protocol']==C.ref(protocol)
  for arm in C.ARMS:
   assert pred['models'][arm]==models[arm]['final_model'];model_sources[arm]=pred['models'][arm]
   with np.load(C.checked(pred['files'][arm]),allow_pickle=False) as f:
    v=f['logits'];conditions[arm]=(np.repeat(v,10,axis=1) if v.shape[1]==1 else v.copy(),f['stay'].copy())
  chunks=[C.OUT/'tasks'/f'images80_{i:02d}' for i in range(4)]
  if mode=='stage400':chunks.extend(C.OUT/'tasks'/f'images400_{i:02d}' for i in range(16))
  arms=all_image_arms();rows=panel;expected=joined['stage80'] if mode=='stage80' else list(range(400))
 same,different=donor_maps(panel);indices={a:[] for a in arms};inodes=set();files=[];pixel_hashes={};all_records={};features={a:[] for a in arms};table=None;min_margin=1.;constant=0
 for root in chunks:
  guard();done=C.read(root/'completion.json')
  assert done['protocol']==C.ref(protocol) and done['arms']==arms and done['frozen_VAR_VQ_unchanged'] and done['actual_native_calls_per_image']==10 and done['engineering_only']==profile
  assert done['status']==('PASS_REAL_ALL6_EEG_OUTPUT_TO_NATIVE_CHAIN_GENERATION_PROFILE' if profile else 'COMPLETE_ALL10_MATCHED_EEG_METHOD_AND_DONOR_IMAGE_CHUNK')
  if profile:assert done['engineering_replayed_images']==6 and done['new_research_images']==0 and done['query_indices']==[0]
  else:
   stage=80 if root.name.startswith('images80_') else 400;part=int(root.name.rsplit('_',1)[1]);selection=joined['stage80'] if stage==80 else [i for i in range(400) if i not in set(joined['stage80'])]
   assert done['stage']==stage and done['part']==part and done['query_indices']==selection[part*20:(part+1)*20] and done['new_research_images']==200
  for arm in arms:
   records=C.read(C.checked(done['records'][arm]));assert len(records)==len(done['query_indices'])
   base,intervention=split_arm(arm);v,z=conditions[base]
   if not profile:
    f=np.load(C.checked(done['features'][arm]),allow_pickle=False);assert f.shape==(20,768) and f.dtype==np.float32 and np.isfinite(f).all()
    assert np.allclose(np.linalg.norm(f.astype(np.float64),axis=-1),1.,atol=2e-5,rtol=0);features[arm].append(done['features'][arm])
   for index,record in zip(done['query_indices'],records):
    row=rows[index];donor=0 if profile else same[index] if intervention=='same' else different[index] if intervention=='different' else index
    assert record['arm']==arm and record['base_arm']==base and record['subject']=='sub-01' and record['query_index']==index
    assert (record['stimulus_id'],record['array_row'])==(row['stimulus_id'],row['array_row'])
    assert record['EEG_prediction_row']==(None if base=='image_prior_full' else donor) and record['EEG_intervention']==intervention and record['model_source']==model_sources[base]
    assert record['latent_seed']==seed(row['stimulus_id'],'latent') and record['token_seed']==seed(row['stimulus_id'],'token')
    assert record['normal_inference_target_or_label_access'] is False and record['labels_only_for_registered_donor_permutations']==(intervention!='correct') and record['generation']==q['inference']
    assert record['prior_logits_SHA256']==hashlib.sha256(v[donor:donor+1].tobytes()).hexdigest() and record['prior_stay_SHA256']==hashlib.sha256(z[donor:donor+1].tobytes()).hexdigest()
    latent=record['latent_rng_SHA256'];assert latent['uniform_draws_per_query']==19 and latent['no_best_of_N'] is True
    assert all(len(latent[k])==64 for k in ('private_latent_rng_before','private_latent_rng_after'))
    chain,margin=independent_chain(v[donor],z[donor],arm,record['latent_uniforms']);assert chain==record['latent_class_chain'];min_margin=min(min_margin,margin)
    trace=record['renderer_trace'];assert trace['scales']==10 and trace['actual_generated_tokens']==680 and trace['CFG']==0 and trace['full_prefix_transformer_calls']==10
    assert trace['one_image_per_query'] and trace['native_cached_sampling_used'] is False and trace['normal_inference_target_access'] is False and trace['extra_learned_parameters']==0
    assert len(trace['records'])==10;prefix=0
    for j,side in enumerate((1,2,3,4,5,6,8,10,13,16)):
     item=trace['records'][j];prefix+=side*side
     assert item['scale']==j and item['classes']==[chain[j]] and item['tokens']==side*side and item['prefix_tokens']==prefix and item['prefix_recomputed_under_current_class']
     assert len(item['sampled_codes_SHA256'])==len(item['logits_SHA256'])==64
    table=record['native_table_SHA256'] if table is None else table;assert record['native_table_SHA256']==table and len(table)==64
    path=C.checked(record['image']);assert path==root/arm/f'{index:04d}.png' and path.resolve().is_relative_to(C.OUT.resolve())
    stat=path.stat();identity=(stat.st_dev,stat.st_ino);assert stat.st_nlink==1 and identity not in inodes;inodes.add(identity)
    with Image.open(path) as image:
     assert image.format=='PNG' and image.mode=='RGB' and image.size==(256,256);raw=np.array(image,copy=True)
    assert raw.dtype==np.uint8 and raw.shape==(256,256,3) and hashlib.sha256(raw.tobytes()).hexdigest()==record['pixel_SHA256']
    # Per-channel spatial variance catches constant purple as well as gray.
    spatial=raw.astype(np.float64).std(axis=(0,1));constant+=int(spatial.max()==0)
    pixel_hashes.setdefault(record['pixel_SHA256'],[]).append([arm,index]);indices[arm].append(index);files.append(record['image'])
   all_records[str(root/arm/'records.json')]=done['records'][arm]
 for arm in arms:assert len(indices[arm])==len(expected) and sorted(indices[arm])==sorted(expected) and len(set(indices[arm]))==len(expected)
 prior80=None
 if mode=='stage400':
  prior80=C.ref(C.OUT/'artifacts_stage80_CPU_audit.json');old=C.read(prior80['path']);assert old['status']=='PASS_ALL_EXPECTED_IMAGES_IDENTITIES_SEEDS_LATENT_CHAINS_AND_FILES' and old['protocol']==C.ref(protocol)
  assert all(all_records[path]==pin for path,pin in old['record_files'].items())
 status='PASS_ALL6_REAL_GENERATION_PROFILE_IMAGES_AND_LATENT_LAW' if profile else 'PASS_ALL_EXPECTED_IMAGES_IDENTITIES_SEEDS_LATENT_CHAINS_AND_FILES'
 target=C.OUT/('generation_profile_CPU_audit.json' if profile else f'artifacts_{mode}_CPU_audit.json')
 C.write(target,dict(status=status,protocol=C.ref(protocol),stage=mode,arms=arms,queries_per_arm=len(expected),images=len(files),record_files=all_records,features=features,image_files=files,preserved_stage80=prior80,minimum_latent_CDF_boundary_margin=min_margin,spatially_constant_images=constant,duplicate_pixel_groups=[v for v in pixel_hashes.values() if len(v)>1],all_samples_retained=True,CPU_replayed_all_latent_inverse_CDF=True,CPU_replayed_native_token_sampling_or_pixels=False,CUDA_initialized=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--mode',choices=['predictions','profile','stage80','stage400'],required=True);v=p.parse_args()
 predictions(v.protocol) if v.mode=='predictions' else artifacts(v.protocol,v.mode)
