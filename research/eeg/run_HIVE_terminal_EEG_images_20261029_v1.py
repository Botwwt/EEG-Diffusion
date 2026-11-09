"""Fixed last-fit EEG terminal condition: paired native images, features/audit.

First three ordinary steps and native sampler are untouched. No query image
features are loaded by the generation path. Donor EEG is a diagnostic only.
"""
import argparse,datetime,gc,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
ARMS=('real_prefix','Gaussian','sameconcept_EEG')
def read(p):return json.loads(Path(p).read_bytes())
def ref(p):
 p=Path(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
def checked(x):
 actual=ref(x['path']);assert actual['sha256']==x['sha256']
 if 'bytes' in x:assert actual['bytes']==x['bytes']
 return Path(x['path'])
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def memory():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
def panel(s):
 d=read(checked(s['D09_completion']));inputs=read(checked(d['inputs']));queries=read(checked(inputs['panel']))['Dcheck'];assert len(queries)==400;return d,inputs,queries
def runtime(s):
 sys.path.insert(0,s['repository_root']);import brainhive_paired_study_contract_20261025_v1 as C
 C.setup_runtime(dict(runtime_manifest=s['runtime_manifest']));import brainhive_native_models_v1 as N
 return N
def conditions(path):
 s=read(path);root=Path(s['output_root']);d,inputs,queries=panel(s);N=runtime(s);import torch,numpy as np
 from brainhive_bounded_component_v1 import BoundedConditionAdapter
 torch.set_num_threads(2);audit=read(checked(s['training_audit_expected']));assert audit['prefix_queries']==13230
 raw=torch.load(checked(s['EEG']),weights_only=True,map_location='cpu',mmap=True);rows=[q['array_row'] for q in queries];eeg=raw[rows][:,46:63,:].contiguous();scope=N.classes();ordinary=BoundedConditionAdapter(predict_risk=False).eval().requires_grad_(False);ostate=torch.load(checked(s['ordinary']),weights_only=True,map_location='cpu');assert ostate['calibration']['selected_strength']==1.;ordinary.load_state_dict(ostate['model'],strict=True);parent=torch.load(checked(s['parent']),weights_only=True,map_location='cpu')['model'];gamma=parent['block.layers.0.norm.weight'];beta=parent['block.layers.0.norm.bias'];models={};arrays={};provenance={}
 for arm in ('real_prefix','Gaussian'):
  final=audit['arms'][arm]['final_model'];state=torch.load(checked(final),weights_only=True,map_location='cpu',mmap=True);assert state['updates']==1654 and state['exposures']==13230 and state['arm']==arm
  model=scope['BrainMLP'](4250,1024,num_layers=1,dropout=.1).eval().requires_grad_(False);model.load_state_dict(state['model'],strict=True);assert torch.equal(model.block.layers[0].norm.weight,gamma) and torch.equal(model.block.layers[0].norm.bias,beta);raw_c=[];corrected=[]
  with torch.no_grad():
   for lo in range(0,400,64):
    c=model(eeg[lo:lo+64]);out=ordinary(c,gamma[None].expand_as(c),beta[None].expand_as(c))[0];raw_c.append(c.numpy());corrected.append(out.numpy())
  arrays[arm+'_raw']=np.concatenate(raw_c);arrays[arm+'_corrected']=np.concatenate(corrected);provenance[arm]=final
 arrays['gamma']=gamma.numpy();arrays['beta']=beta.numpy();p=root/'CPU_conditions400.npz';np.savez_compressed(p,**arrays);assert not torch.cuda.is_initialized();write(root/'CPU_conditions_completion.json',dict(status='ALL400_BOTH_FINAL_EEG_AND_FROZEN_ORDINARY_CPU_FORWARD_COMPLETE',conditions=ref(p),final_models=provenance,EEG=s['EEG'],query_rows=rows,queries=400,teacher_F_not_loaded=True,query_image_not_loaded=True,CUDA_initialized=False))
def gpu(path,mode,lo,hi):
 s=read(path);assert (lo,hi) in ((0,80),(80,240),(240,400));assert mode in ('generate','features');root=Path(s['output_root']);folder=root/f'{mode}_{lo:03d}_{hi:03d}';folder.mkdir();start=time.monotonic();base=memory();phase='RUNTIME'
 for x in s['source_pins']:checked(x)
 assert 'run_gpu_job_v2.py' in (Path('/proc')/str(os.getppid())/'cmdline').read_text();assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==s['GPU_UUID'];assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();d,inputs,queries=panel(s);N=runtime(s)
 import torch,numpy as np
 from PIL import Image
 torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(min(1.,float(20<<30)/torch.cuda.get_device_properties(0).total_memory))
 def budget():assert time.monotonic()-start<800 and memory()-base<10<<30 and torch.cuda.max_memory_allocated()<20<<30 and shutil.disk_usage(root).free>3<<30
 try:
  if mode=='features':
   import image_evaluation as E
   checked(s['DINO_weight']);assert E.DINO_SHA256==s['DINO_weight']['sha256'];model=E.dino_model();pieces={}
   for arm in ARMS:
    rows=[read(root/'records'/f'{i:03d}_{arm}.json') for i in range(lo,hi)];paths=[]
    for i,r in zip(range(lo,hi),rows):assert r['query_id']==queries[i]['query_id'] and r['query_image_used_for_condition'] is False;paths.append(checked(dict(path=r['generated_path'],bytes=Path(r['generated_path']).stat().st_size,sha256=r['generated_sha256'])))
    parts=[]
    for begin in range(0,len(paths),32):budget();parts.append(E.dino_features(paths[begin:begin+32],model,batch_size=32,real_images=False).float().numpy())
    feature=np.concatenate(parts);assert feature.shape==(hi-lo,768);p=root/f'features_{lo:03d}_{hi:03d}_{arm}.npy';np.save(p,feature,allow_pickle=False);pieces[arm]=ref(p)
   write(folder/'completion.json',dict(status='COMPLETE_FROZEN_DINO_FEATURE_PART',start=lo,stop=hi,features=pieces,seconds=time.monotonic()-start));return
  phase='EXACT_NATIVE_LOAD';from brainhive_component_generator_v1 import load_native
  from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
  from generate_BrainHIVE_neural_clean_fastscreen_20261027_v1 import save_image
  sys.path.insert(0,s['native_switch_directory']);from run_HIVE_terminal_visual_probe_20261029_v1 import native_switch,tensor_sha
  from brainhive_bounded_component_v1 import BoundedConditionAdapter
  (folder/'loader').mkdir();observer=ConvertedWeightAudit(torch,folder/'loader');observer.install()
  try:pipe,A,P,prompts,host=load_native(training=False)
  finally:observer.restore()
  loading=observer.complete();assert host==inputs['expected_host'];audit=read(checked(s['training_audit_expected']));cpu_done=read(root/'CPU_conditions_completion.json');cpu=np.load(checked(cpu_done['conditions']),allow_pickle=False);baseline=np.load(checked(d['check_conditions']),allow_pickle=False);assert baseline.shape==(400,3,1024)
  ordinary=BoundedConditionAdapter(predict_risk=False).cuda().eval().requires_grad_(False);ordinary.load_state_dict(torch.load(checked(s['ordinary']),weights_only=True,map_location='cpu')['model'],strict=True);scope=N.classes();raw=torch.load(checked(s['EEG']),weights_only=True,map_location='cpu',mmap=True);indices=[q['array_row'] for q in queries];eeg=raw[indices][:,46:63,:].contiguous();gamma=torch.from_numpy(cpu['gamma'].copy()).cuda();beta=torch.from_numpy(cpu['beta'].copy()).cuda();all_conditions=[];errors={};final_models={}
  for arm in ('real_prefix','Gaussian'):
   final=cpu_done['final_models'][arm];assert final==audit['arms'][arm]['final_model'];state=torch.load(checked(final),weights_only=True,map_location='cpu',mmap=True);model=scope['BrainMLP'](4250,1024,num_layers=1,dropout=.1).cuda().eval().requires_grad_(False);model.load_state_dict(state['model'],strict=True);values=[]
   with torch.no_grad():
    for begin in range(0,400,64):
     c=model(eeg[begin:begin+64].cuda());out=ordinary(c,gamma[None].expand_as(c),beta[None].expand_as(c))[0];values.append(out.cpu().numpy())
   value=np.concatenate(values);delta=value.astype(np.float64)-cpu[arm+'_corrected'].astype(np.float64);absolute=float(np.abs(delta).max());relative=float(np.linalg.norm(delta)/np.linalg.norm(cpu[arm+'_corrected'].astype(np.float64)));assert absolute<1e-3 and relative<1e-5;errors[arm]=dict(maximum_absolute=absolute,relative_L2=relative);all_conditions.append(value);final_models[arm]=final;del model;gc.collect();torch.cuda.empty_cache()
  condition=np.stack(all_conditions,axis=1);p=root/'GPU_conditions400.npy'
  if lo==0:assert not p.exists();np.save(p,condition,allow_pickle=False);write(root/'all400_CPU_GPU_condition_check.json',dict(status='ALL800_FINAL_CONDITIONS_CPU_GPU_NUMERICALLY_VERIFIED',errors=errors,conditions=ref(p),final_models=final_models,normal_query_target_access=False))
  else:assert np.array_equal(condition,np.load(p,allow_pickle=False))
  old=[json.loads(line) for line in checked(d['check_manifest']).read_text().splitlines()];refs={(r['query_id'],r['arm']):r for r in old};frozen={}
  def digest(module):
   h=hashlib.sha256()
   for n,v in module.state_dict().items():h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
   return h.hexdigest()
  modules=dict(ordinary=ordinary,A=A,P=P,UNet=pipe.unet,VAE=pipe.vae);frozen={k:digest(v) for k,v in modules.items()}
  for name in ('records','images','ordinary_replays'):(root/name).mkdir(exist_ok=True)
  phase='PAIRED_NATIVE_LAST_STEP_EEG_IMAGES'
  for i in range(lo,hi):
   budget();q=queries[i];donor=s['same_concept_donor_indices'][i];assert donor!=i and queries[donor]['concept_id']==q['concept_id'];reference=refs[(q['query_id'],'ordinary')]
   with torch.no_grad():
    base_tokens=P(A(torch.from_numpy(baseline[i:i+1,1].copy()).cuda().half()));image,proof,states=native_switch(pipe,prompts,base_tokens,base_tokens,q['image_seed'],torch)
   with Image.open(reference['generated_path']) as old_image:assert np.array_equal(np.asarray(image),np.asarray(old_image.convert('RGB')))
   assert ref(reference['generated_path'])['sha256']==reference['generated_sha256']
   for key in ('initial_noise_latents_sha256','initial_UNet_input_sha256','ancestral_noise_observations'):assert proof[key]==reference[key]
   write(root/'ordinary_replays'/f'{i:03d}.json',dict(**q,reference=reference,proof=proof,pixel_exact=True))
   for arm in ARMS:
    source=i if arm!='sameconcept_EEG' else donor;column=1 if arm=='Gaussian' else 0;last=torch.from_numpy(condition[source:source+1,column].copy()).cuda();checkpoint=final_models['Gaussian' if arm=='Gaussian' else 'real_prefix']
    with torch.no_grad():last_tokens=P(A(last.half()));image,current,end=native_switch(pipe,prompts,base_tokens,last_tokens,q['image_seed'],torch)
    for key in ('initial_noise_latents_sha256','initial_UNet_input_sha256','ancestral_noise_observations'):assert current[key]==proof[key]
    assert [x['UNet_input_sha256'] for x in current['all_step_observations']]==[x['UNet_input_sha256'] for x in proof['all_step_observations']];assert [x['IP_tokens_sha256'] for x in current['all_step_observations'][:3]]==[x['IP_tokens_sha256'] for x in proof['all_step_observations'][:3]];assert torch.equal(states['prefix'],end['prefix'])
    saved=save_image(root/'images'/f'{i:03d}_{arm}.webp',image);write(root/'records'/f'{i:03d}_{arm}.json',dict(**q,query_index=i,arm=arm,last_EEG_source_query=queries[source],final_EEG_checkpoint=checkpoint,last_condition_sha256=tensor_sha(last),last_FP16_condition_sha256=tensor_sha(last.half()),query_image_used_for_condition=False,concept_label_used_for_normal_method=False,diagnostic_donor_intervention=arm=='sameconcept_EEG',training_protocol=s['training_protocol'],image_protocol=ref(path),**current,**saved))
   if (i-lo+1)%10==0:print(json.dumps(dict(queries=i-lo+1,start=lo,stop=hi,images=3*(i-lo+1),seconds=time.monotonic()-start)),flush=True)
  assert frozen=={k:digest(v) for k,v in modules.items()};checked(s['ordinary']);budget();write(folder/'completion.json',dict(status='COMPLETE_NORMAL_EEG_NATIVE_LAST_STEP_IMAGE_PART',start=lo,stop=hi,queries=hi-lo,research_images=3*(hi-lo),ordinary_pixel_exact_replays=hi-lo,all_first4_UNet_inputs_and_all_noises_same=True,all_frozen_weights_unchanged=True,loader=loading,host=host,final_models=final_models,seconds=time.monotonic()-start,peak_GPU_allocated=torch.cuda.max_memory_allocated(),extra_CPU_peak=memory()-base))
 except BaseException as error:write(folder/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-start));raise
def score(path,count):
 s=read(path);root=Path(s['output_root']);sys.path.insert(0,s['repository_root']);import numpy as np
 import score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v2 as S
 d,inputs,queries=panel(s);targets=np.load(checked(d['check_target_features']),allow_pickle=False).astype(np.float64);base=np.load(checked(d['check_generated_features']),allow_pickle=False);metrics={'ordinary':S.metric_arrays(base[:count,1].astype(np.float64),targets,queries[:count],queries)};features={}
 for arm in ARMS:
  pieces=[np.load(root/f'features_000_080_{arm}.npy',allow_pickle=False)]
  if count==400:pieces.extend(np.load(root/f'features_{lo:03d}_{hi:03d}_{arm}.npy',allow_pickle=False) for lo,hi in ((80,240),(240,400)))
  feature=np.concatenate(pieces);assert feature.shape==(count,768);features[arm]=feature;metrics[arm]=S.metric_arrays(feature.astype(np.float64),targets,queries[:count],queries)
 pairs={label:S.compare(metrics[a],metrics[b],queries[:count]) for label,a,b in [('real_minus_ordinary','real_prefix','ordinary'),('Gaussian_minus_ordinary','Gaussian','ordinary'),('real_minus_Gaussian','real_prefix','Gaussian'),('real_minus_donor','real_prefix','sameconcept_EEG')]};decision=S.decision(pairs['real_minus_ordinary'],str(count));p=root/f'metrics{count}.npz';np.savez_compressed(p,**{arm+'__'+key:value for arm,m in metrics.items() for key,value in m.items()});np.savez_compressed(root/f'all_features{count}.npz',**features)
 result=dict(status='COMPLETE_FIXED_GALLERY_TERMINAL_EEG_IMAGE_SCORE',queries=count,gallery_targets=400,means={arm:{key:float(value.mean()) for key,value in m.items()} for arm,m in metrics.items()},contrasts=pairs,primary_decision=decision,primary_candidate='real_prefix',no_control_selected_as_winner=True,all_control_arms_expand_with_primary_candidate=True,metrics=ref(p),normal_EEG_target_access=False,exposed_development_panel=True,single_subject=True,single_training_seed=True,independent_confirmation=False,SOTA=False)
 if count==400:
  earlier=read(root/'score80.json');assert earlier['primary_decision']['action']=='EXPAND400'
  for arm in ARMS:assert np.array_equal(features[arm][:80],np.load(root/f'features_000_080_{arm}.npy',allow_pickle=False))
  m=pairs['real_minus_Gaussian'];e=pairs['real_minus_donor'];result['training_state_specific_support']=m['DINO2AFC']['difference']>0 and m['DINO2AFC']['descriptive_concept_bootstrap95'][0]>0 and m['within_concept10_top1']['difference']>=0;result['paired_last_EEG_stimulus_support']=e['within_concept10_top1']['difference']>0 and e['within_concept10_top1']['descriptive_concept_bootstrap95'][0]>0
 write(root/f'score{count}.json',result);print(json.dumps(result))
def audit(path,count):
 s=read(path);root=Path(s['output_root']);d,inputs,queries=panel(s);import numpy as np
 from PIL import Image
 score=read(root/f'score{count}.json');condition=np.load(root/'GPU_conditions400.npy',allow_pickle=False);paths=set();inodes=set()
 for i,q in enumerate(queries[:count]):
  old=read(root/'ordinary_replays'/f'{i:03d}.json');assert old['pixel_exact'];reference=old['reference'];assert ref(reference['generated_path'])['sha256']==reference['generated_sha256']
  for arm in ARMS:
   r=read(root/'records'/f'{i:03d}_{arm}.json');assert all(r[k]==v for k,v in q.items());assert r['UNet_NFE']==4 and r['native_IP_K_calls']==r['native_IP_V_calls']==280 and r['query_image_used_for_condition'] is False
   source=s['same_concept_donor_indices'][i] if arm=='sameconcept_EEG' else i;assert r['last_EEG_source_query']==queries[source];column=1 if arm=='Gaussian' else 0;c=condition[source:source+1,column];assert hashlib.sha256(c.tobytes()).hexdigest()==r['last_condition_sha256'];assert hashlib.sha256(c.astype(np.float16).tobytes()).hexdigest()==r['last_FP16_condition_sha256']
   for key in ('initial_noise_latents_sha256','initial_UNet_input_sha256','ancestral_noise_observations'):assert r[key]==old['proof'][key]
   assert [x['UNet_input_sha256'] for x in r['all_step_observations']]==[x['UNet_input_sha256'] for x in old['proof']['all_step_observations']];assert [x['IP_tokens_sha256'] for x in r['all_step_observations'][:3]]==[x['IP_tokens_sha256'] for x in old['proof']['all_step_observations'][:3]]
   p=Path(r['generated_path']);assert p.resolve().is_relative_to(root.resolve()) and str(p) not in paths;paths.add(str(p));st=p.stat();ino=(st.st_dev,st.st_ino);assert ino not in inodes;inodes.add(ino);assert ref(p)['sha256']==r['generated_sha256']
   with Image.open(p) as image:assert image.size==(512,512);assert hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()==r['generated_pixel_sha256']
 targets=np.load(checked(d['check_target_features']),allow_pickle=False).astype(np.float64);targets/=np.linalg.norm(targets,axis=1,keepdims=True);base=np.load(checked(d['check_generated_features']),allow_pickle=False);features=np.load(root/f'all_features{count}.npz',allow_pickle=False);actual=np.load(root/f'metrics{count}.npz',allow_pickle=False);metrics={};errors=[]
 for arm,value in [('ordinary',base[:count,1])]+[(arm,features[arm]) for arm in ARMS]:
  x=value.astype(np.float64);x/=np.linalg.norm(x,axis=1,keepdims=True);sim=x@targets.T;m={k:np.empty(count) for k in score['means'][arm]}
  for i in range(count):
   row=sim[i];v=row[i];group=[j for j,q in enumerate(queries) if q['concept_id']==queries[i]['concept_id']];local=row[group];m['target_cosine'][i]=v;m['DINO2AFC'][i]=((row<v).sum()+.5*((row==v).sum()-1))/399;m['gallery_top1'][i]=row.argmax()==i;m['within_concept10_top1'][i]=group[int(local.argmax())]==i;m['within_concept10_two_afc'][i]=((local<v).sum()+.5*((local==v).sum()-1))/9
  for key,value in m.items():errors.append(float(np.abs(value-actual[arm+'__'+key]).max()));errors.append(abs(float(value.mean())-score['means'][arm][key]))
  metrics[arm]=m
 groups=np.array([[i for i in range(count) if queries[i]['concept_id']==c] for c in sorted({q['concept_id'] for q in queries[:count]})]);n=len(groups);draws=np.random.default_rng(1729).integers(0,n,size=(1000,n));weights=np.array([np.bincount(row,minlength=n) for row in draws])/n
 for label,a,b in [('real_minus_ordinary','real_prefix','ordinary'),('Gaussian_minus_ordinary','Gaussian','ordinary'),('real_minus_Gaussian','real_prefix','Gaussian'),('real_minus_donor','real_prefix','sameconcept_EEG')]:
  for key in metrics[a]:
   delta=metrics[a][key]-metrics[b][key];ci=np.quantile(np.einsum('bc,c->b',weights,delta[groups].mean(1)),[.025,.975]);target=score['contrasts'][label][key];errors.append(abs(float(delta.mean())-target['difference']));errors.extend(np.abs(ci-target['descriptive_concept_bootstrap95']).tolist())
 assert max(errors)<1e-12;np.savez_compressed(root/f'independent_metrics{count}.npz',**{a+'__'+k:v for a,m in metrics.items() for k,v in m.items()});write(root/f'independent_images_and_statistics{count}_CPU_audit.json',dict(status='ALL_REGISTERED_IMAGES_NOISE_CONDITIONS_AND_INDEPENDENT_STATISTICS_VERIFIED',queries=count,new_images=3*count,ordinary_exact_replays=count,unique_paths=len(paths),unique_inodes=len(inodes),maximum_statistical_error=max(errors),statistical_error_checks=len(errors),independent_vision_reencoding=False,all_query_denominators_retained=True))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--mode',choices=['conditions','generate','features','score','audit'],required=True);p.add_argument('--lo',type=int,default=0);p.add_argument('--hi',type=int,default=80);p.add_argument('--count',type=int,choices=[80,400],default=80);a=p.parse_args()
 if a.mode=='conditions':conditions(a.protocol)
 elif a.mode in ('generate','features'):gpu(a.protocol,a.mode,a.lo,a.hi)
 elif a.mode=='score':score(a.protocol,a.count)
 else:audit(a.protocol,a.count)
