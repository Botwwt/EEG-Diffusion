"""EEG-only prediction, actual interface profile, and matched complete images."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import argparse,datetime,gc,hashlib,importlib.util,json,resource,subprocess,time,traceback
from pathlib import Path
import EEG_complete_native_study_contract_CPU_20261102_v1 as C

def seed(stimulus,kind):return int.from_bytes(hashlib.sha256(f'1729|EEG-native-marginal-v2|{kind}|sub-01|{stimulus}|R4'.encode()).digest()[:8],'little')%(2**63-1)
def image_arms():return list(C.ARMS)+[a+s for a in C.CANDIDATES for s in ('_same','_different')]
def role(arm):
 for suffix in ('_same','_different'):
  if arm.endswith(suffix):return arm[:-len(suffix)],suffix[1:]
 return arm,'correct'
def all_final(q):
 result={}
 for arm in C.ARMS:
  root=C.OUT/'fits'/arm;done=C.read(root/'completion.json');cpu=C.read(root/'fit10_CPU_audit.json')
  assert done['status']=='COMPLETE_FIXED10_FULL_T_EEG_ARM' and done['arm']==cpu['arm']==arm and done['epoch']==cpu['epoch']==10 and done['updates']==cpu['updates']==2070 and cpu['status']=='PASS_ALL_FIXED_EEG_EPOCH_STATE_ADAM_COVERAGE'
  assert Path(done['final_model']['path'])==root/'model_final.pt'
  assert cpu['protocol']==done['protocol']==C.ref(C.CODE/'protocol.json') and cpu['final_model']==done['final_model'];C.checked(done['final_model']);result[arm]=done
 return result

def run(protocol,mode,job,stage=80,part=0):
 q=C.setup(protocol,formal=mode!='profile');C.resources(q,starting=True);base,a,image,stats=C.source_inputs(q,binary=False);repo=Path(q['repository_root'])
 receipt=C.read(repo/'audit'/(job+'_job.json'));assert receipt['status']=='RUNNING' and receipt['pid']==os.getpid() and receipt['max_seconds']==900
 assert 'run_gpu_job_v2.py' in Path('/proc',str(os.getppid()),'cmdline').read_bytes().decode().replace('\0',' ')
 assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==C.GPU
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 work=C.OUT/'tasks'/('generation_profile' if mode=='profile' else 'predict_final400' if mode=='predict' else f'images{stage}_{part:02d}');work.mkdir(parents=True);start=time.monotonic();base_RSS=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
 try:
  C.roots();import numpy as np,torch
  from PIL import Image
  import EEG_complete_native_runtime_20261102_v1 as R
  from EEG_complete_native_study_core_20261102_v1 import sampling_parameters
  torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  last=0.
  def guard():
   nonlocal last
   assert time.monotonic()-start<q['GPU_soft_seconds'] and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-base_RSS<q['maximum_extra_CPU_bytes'] and torch.cuda.max_memory_allocated()<q['maximum_GPU_bytes']
   if time.monotonic()-last>10:C.resources(q);last=time.monotonic()
  inputs=C.read(C.OUT/'inputs/completion.json');assert inputs['status']=='PASS_COLD_COMPLETE_T13230_DCHECK400_REAL_EEG_INPUT_JOIN_TRANSFORM'
  joined=C.read(C.checked(inputs['identities']));panel=C.read(C.checked(image['panel']));P=panel['queries'];assert len(P)==400
  if mode=='predict':
   finals=all_final(q);x=R.inputs(q,'Dcheck');channels=R.channels(q);files={};model_sources={}
   for arm in C.ARMS:
    guard();p=R.model(q,arm,'cuda');done=finals[arm];state=torch.load(C.checked(done['final_model']),map_location='cpu',weights_only=True)
    assert state['purpose']=='FORMAL_FIXED_FULL_T_FINAL_ONLY' and state['arm']==arm and state['updates']==2070 and state['protocol']==C.ref(protocol);p.load_state_dict(state['model'],strict=True);del state
    logits=[];stay=[]
    with torch.inference_mode():
     for lo in range(0,400,64):
      v,z=p(torch.from_numpy(np.array(x[lo:lo+64])).cuda(),input_chans=channels);assert torch.isfinite(v).all() and torch.isfinite(z).all();logits.append(v[:,0:1].cpu().numpy() if p.family=='global' else v.cpu().numpy());stay.append(z.cpu().numpy())
    v=np.concatenate(logits);z=np.concatenate(stay);assert v.shape==(400,1 if p.family=='global' else 10,1000) and z.shape==(400,9)
    path=work/(arm+'.npz');np.savez_compressed(path,logits=v,stay=z);files[arm]=C.ref(path);model_sources[arm]=done['final_model'];del p;gc.collect();torch.cuda.empty_cache()
   C.write(work/'completion.json',dict(status='COMPLETE_ALL6_FINAL_EEG_ONLY_DCHECK400_PREDICTIONS',protocol=C.ref(protocol),files=files,models=model_sources,EEG_input=inputs['files']['Dcheck'],query_images_or_labels_read_by_predictor=False,seconds=time.monotonic()-start,new_research_images=0));return
  conditions={};model_sources={}
  if mode=='profile':
   audit=C.read(C.OUT/'all6_profiles_CPU_audit.json');assert audit['status']=='PASS_ALL6_REAL_PROFILE_GRADIENTS_AND_SERIALIZED_RESUME_STATES'
   for arm in C.ARMS:
    done=C.read(C.OUT/'tasks'/(arm+'_profile')/'completion.json')
    with np.load(C.checked(done['traces'][2])) as f:conditions[arm]=(f['logits'][:1].copy(),f['stay'][:1].copy())
    model_sources[arm]=dict(engineering_trace=done['traces'][2],purpose='BEFORE_TEMPORARY_UPDATE3_NOT_A_METHOD_MODEL')
   indices=[0];arms=list(C.ARMS);query_rows=joined['T_rows']
  else:
   assert mode=='images' and stage in (80,400);all_final(q)
   pred=C.read(C.OUT/'tasks/predict_final400/completion.json');cpu=C.read(C.OUT/'final_predictions_CPU_audit.json')
   assert pred['status']=='COMPLETE_ALL6_FINAL_EEG_ONLY_DCHECK400_PREDICTIONS' and cpu['status']=='PASS_ALL6_FINAL400_CPU_FORWARD_AND_NOEEG_INDEPENDENCE'
   for arm in C.ARMS:
    with np.load(C.checked(pred['files'][arm])) as f:
     v=f['logits'];conditions[arm]=(np.repeat(v,10,axis=1) if v.shape[1]==1 else v.copy(),f['stay'].copy())
    model_sources[arm]=pred['models'][arm]
   first=joined['stage80'];remaining=[i for i in range(400) if i not in set(first)];selected=first if stage==80 else remaining
   assert len(selected)==(80 if stage==80 else 320) and 0<=part<len(selected)//20;indices=selected[part*20:(part+1)*20];arms=image_arms();query_rows=P
   if stage==400:
    screen=C.read(C.OUT/'screens/stage80/summary.json');assert screen['expand_any'] and screen['protocol']==C.ref(protocol)
  from EEG_generative_backend_loader_20261029_v1 import build_and_load
  from VAR_declared_FP16_kernel_20261102_v1 import DeclaredFP16Native
  import EEG_multiscale_native_kernel_DRAFT_20261101_v1 as K
  from profile_VAR_multiscale_native_GPU_20261101_v1 import digest
  vae,d,backend=build_and_load(a['author_source'],a['vendor'],C.checked(a['public_VQ']),C.checked(a['public_VAR']),seed=1729)
  public=torch.load(a['public_VAR']['path'],map_location='cpu',weights_only=True,mmap=True);native=DeclaredFP16Native(d,public['class_emb.weight'].clone()).eval().requires_grad_(False);del public
  before=dict(VAR=digest(native),VQ=digest(vae));native.cuda();vae.cuda().eval().requires_grad_(False);table_sha=hashlib.sha256(native.native_class_embedding.cpu().numpy().tobytes()).hexdigest();torch.cuda.reset_peak_memory_stats()
  groups={}
  for i,row in enumerate(P):groups.setdefault(row['concept_id'],[]).append(i)
  keys=sorted(groups);same={i:groups[k][(j+1)%10] for k in keys for j,i in enumerate(groups[k])};different={i:groups[keys[(n+1)%40]][j] for n,k in enumerate(keys) for j,i in enumerate(groups[k])}
  assert sorted(same.values())==sorted(different.values())==list(range(400))
  records={};paths={};profile_replays=0
  for arm in arms:
   directory=work/arm;directory.mkdir();records[arm]=[];paths[arm]=[];basearm,intervention=role(arm)
   for index in indices:
    guard();row=query_rows[index];source=0 if mode=='profile' else same[index] if intervention=='same' else different[index] if intervention=='different' else index
    v,z=conditions[basearm];v=torch.from_numpy(v[source:source+1]).cuda();z=torch.from_numpy(z[source:source+1]).cuda();logits,rho=sampling_parameters(v,z,basearm)
    cs=seed(row['stimulus_id'],'latent');ts=seed(row['stimulus_id'],'token');rng=torch.Generator(device='cuda').manual_seed(cs)
    with torch.inference_mode():
     chain,latent=K.draw_class_chain(logits,rho,generator=rng);pixels,trace=K.generate_class_chain(native,chain,image_seed=ts,top_k=0)
    assert pixels.shape==(1,3,256,256) and torch.isfinite(pixels).all();raw=pixels[0].mul(255).add(.5).clamp(0,255).byte().permute(1,2,0).cpu().numpy()
    if mode=='profile':
     with torch.inference_mode():
      chain2,latent2=K.draw_class_chain(logits,rho,generator=torch.Generator(device='cuda').manual_seed(cs));replay,_=K.generate_class_chain(native,chain2,image_seed=ts,top_k=0)
     assert torch.equal(chain,chain2) and torch.equal(pixels,replay);profile_replays+=1
    path=directory/f'{index:04d}.png';assert not path.exists();Image.fromarray(raw).save(path)
    meta=dict(arm=arm,base_arm=basearm,subject='sub-01',query_index=index,stimulus_id=row['stimulus_id'],array_row=row['array_row'],EEG_prediction_row=None if basearm=='image_prior_full' else source,EEG_intervention=intervention,model_source=model_sources[basearm],latent_seed=cs,token_seed=ts,latent_class_chain=chain.cpu().tolist()[0],latent_uniforms={k:value.cpu().tolist() for k,value in latent.items() if isinstance(value,torch.Tensor)},latent_rng_SHA256={k:value for k,value in latent.items() if not isinstance(value,torch.Tensor)},prior_logits_SHA256=hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest(),prior_stay_SHA256=hashlib.sha256(z.cpu().numpy().tobytes()).hexdigest(),image=C.ref(path),pixel_SHA256=hashlib.sha256(raw.tobytes()).hexdigest(),native_table_SHA256=table_sha,renderer_trace=trace,normal_inference_target_or_label_access=False,labels_only_for_registered_donor_permutations=intervention!='correct',generation=q['inference'])
    records[arm].append(meta);paths[arm].append(path)
   C.write(directory/'records.json',records[arm])
  assert before==dict(VAR=digest(native),VQ=digest(vae));del native,vae,d,conditions;gc.collect();torch.cuda.empty_cache()
  features={}
  if mode=='images':
   evaluator=importlib.util.spec_from_file_location('complete_native_locked_image_evaluation',C.checked(image['image_evaluation']));E=importlib.util.module_from_spec(evaluator);evaluator.loader.exec_module(E)
   assert C.ref(E.__file__)==image['image_evaluation'] and E.DINO_SHA256==image['DINO_weight']['sha256'];C.checked(image['DINO_weight']);critic=E.dino_model()
   for arm in arms:
    guard();f=E.dino_features(paths[arm],critic,batch_size=20,real_images=False).numpy().astype(np.float32);assert f.shape==(20,768) and np.isfinite(f).all();path=work/arm/'features.npy';np.save(path,f,allow_pickle=False);features[arm]=C.ref(path)
  guard();C.write(work/'completion.json',dict(status='PASS_REAL_ALL6_EEG_OUTPUT_TO_NATIVE_CHAIN_GENERATION_PROFILE' if mode=='profile' else 'COMPLETE_ALL10_MATCHED_EEG_METHOD_AND_DONOR_IMAGE_CHUNK',protocol=C.ref(protocol),stage=stage if mode=='images' else None,part=part if mode=='images' else None,query_indices=indices,arms=arms,records={arm:C.ref(work/arm/'records.json') for arm in arms},features=features,source_predictions=None if mode=='profile' else C.ref(C.OUT/'tasks/predict_final400/completion.json'),frozen_VAR_VQ_unchanged=True,native_state_digests=before,backend=backend,actual_native_calls_per_image=10,engineering_replayed_images=profile_replays,engineering_only=mode=='profile',new_research_images=len(indices)*len(arms) if mode=='images' else 0,seconds=time.monotonic()-start,GPU_peak=torch.cuda.max_memory_allocated(),extra_CPU_peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-base_RSS))
 except BaseException as e:C.write(work/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),no_retry=True));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--mode',choices=['profile','predict','images'],required=True);p.add_argument('--job-name',required=True);p.add_argument('--stage',type=int,default=80);p.add_argument('--part',type=int,default=0);v=p.parse_args();run(v.protocol,v.mode,v.job_name,v.stage,v.part)
