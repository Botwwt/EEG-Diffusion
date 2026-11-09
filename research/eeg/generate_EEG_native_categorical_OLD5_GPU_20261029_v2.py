"""Continue the predeclared five arms on OLD, reusing exact fixed EEG logits."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import argparse,datetime,gc,hashlib,json,shutil,subprocess,sys,time,traceback
from pathlib import Path
from generate_EEG_native_marginal_images_GPU_20261029_v2 import sha,ref,check,write,peak,seed,digest

ARMS=['coupled','same_donor','different_donor','visual','uniform']


def run(sp,lo,hi,job):
 s=json.loads(Path(sp).read_bytes())
 assert s['status']=='REGISTERED_CATEGORICAL_OLD_ONLY_EXACT_LOGITS_STAGE400_v2'
 assert s['execution_host']=='old' and s['arms']==ARMS and 80<=lo<hi<=400 and lo%20==0 and hi==lo+20
 for x in s['source_pins']:check(x)
 decision=json.loads(check(s['stage80_decision']).read_bytes())
 assert decision['status']=='COMPLETE_ALL5_CATEGORICAL_FIXED80_IMAGE_SCREEN' and decision['decision']=='EXPAND400'
 assert 'run_gpu_job_v2.py' in (Path('/proc')/str(os.getppid())/'cmdline').read_bytes().decode()
 assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==s['GPU_UUID']
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 ledger=json.loads((Path(s['repository_root'])/'audit'/(job+'_job.json')).read_bytes())
 assert ledger['status']=='RUNNING' and ledger['pid']==os.getpid() and float(ledger['max_seconds'])==900
 root=Path(s['output_root']);root.mkdir(exist_ok=True);out=root/f'chunk_{lo:04d}_{hi:04d}';out.mkdir();started=time.monotonic();base=peak();phase='PINNED_SOURCE_FITS_AND_CACHED_EEG_ONLY_LOGITS'
 try:
  import numpy as np
  import torch
  from PIL import Image
  from EEG_native_categorical_core_20261029_v1 import draw_native_class
  from EEG_native_conditional_variational_core_20261029_v1 import NativeConditionDecoder
  from EEG_generative_backend_loader_20261029_v1 import build_and_load
  panel=json.loads(check(s['panel']).read_bytes());queries=panel['queries'];assert len(queries)==400
  source={};records80={};features80={};model_sha={}
  for host in ('old','new'):
   binding=s['source_bindings'][host];fit=json.loads(check(binding['fit_completion']).read_bytes());cpu=json.loads(check(binding['fit_CPU_audit']).read_bytes());audit=json.loads(check(binding['artifact_audit']).read_bytes())
   assert fit['status']=='COMPLETE_FIXED10_NATIVE_CATEGORICAL_FULL_T_FIT' and fit['epochs']==10 and fit['updates']==2070
   assert cpu['status']=='PASS_NATIVE_CATEGORICAL_FULL10_MODEL_ADAM_COVERAGE_AND_RNG_CPU' and cpu['model']==fit['final_model']
   assert audit['status']=='PASS_READ_ONLY_ALL_EXISTING80_IMAGE_PIXELS_IDENTITIES_SEEDS_EEG_ROWS_AND_FEATURES' and audit['all4_cached400_logits_bit_identical']
   arr=np.concatenate([np.load(check(x),allow_pickle=False) for x in binding['logit_parts']])
   assert arr.shape==(400,1000) and arr.dtype==np.float32 and np.isfinite(arr).all()
   assert hashlib.sha256(arr.tobytes()).hexdigest()==binding['logits_raw_sha256'];source[host]=torch.from_numpy(arr).cuda();model_sha[host]=fit['final_model']['sha256']
   for arm in binding['arms']:
    records80[arm]=json.loads(check(binding['records80'][arm]).read_bytes());features80[arm]=np.load(check(binding['features80'][arm]),allow_pickle=False)
    assert len(records80[arm])==80 and features80[arm].shape==(80,768)
  base_spec=json.loads(check(s['OLD_base_EEG_spec']).read_bytes());sys.path.insert(0,base_spec['training_code']);import AR_anchored_training_contract_CPU_20261029_v1 as C
  a,_=C.load_admission(base_spec['training_admission']);assert a['host_role']=='old';C.validate_sources(a);C.verify_protected_files(a);sys.path.insert(0,a['vendor'])
  torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  groups={}
  for i,q in enumerate(queries):groups.setdefault(q['concept_id'],[]).append(i)
  keys=sorted(groups);assert len(keys)==40 and {len(v) for v in groups.values()}=={10}
  same={i:groups[k][(j+1)%10] for k in keys for j,i in enumerate(groups[k])}
  different={i:groups[keys[(n+1)%40]][j] for n,k in enumerate(keys) for j,i in enumerate(groups[k])}
  phase='UNCHANGED_NATIVE_VAR_VQ';vae,d,backend=build_and_load(a['author_source'],a['vendor'],check(a['public_VQ']),check(a['public_VAR']),seed=1729)
  public=torch.load(check(a['public_VAR']),weights_only=True,map_location='cpu',mmap=True);classes=public['class_emb.weight'].clone();del public
  before=digest({k:v for k,v in d.state_dict().items() if not k.startswith('embed_proj.')});vq_before=digest(vae.state_dict())
  native=NativeConditionDecoder(d,classes).cuda().eval();vae.cuda().eval().requires_grad_(False);d.rng=torch.Generator(device='cuda');table=classes[:1000].cuda();table_sha=hashlib.sha256(classes.numpy().tobytes()).hexdigest()
  assert table_sha==s['native_class_table_SHA256']
  def guard():
   assert time.monotonic()-started<800 and peak()-base<10<<30 and torch.cuda.max_memory_allocated()<20<<30 and shutil.disk_usage(out).free>2<<30
  def image(arm,index):
   q=queries[index];host='old' if arm in ('coupled','same_donor','different_donor') else 'new'
   row=same[index] if arm=='same_donor' else different[index] if arm=='different_donor' else index
   logits=torch.zeros((1,1000),device='cuda') if arm=='uniform' else source[host][row:row+1]
   cs=seed(q['stimulus_id'],'latent');ts=seed(q['stimulus_id'],'token');k=int(draw_native_class(logits,generator=torch.Generator(device='cuda').manual_seed(cs))[0])
   with torch.inference_mode():x=native.generate(table[k:k+1],seed=ts,guidance=0.,top_k=0)
   assert x.shape==(1,3,256,256) and torch.isfinite(x).all()
   pix=x[0].mul(255).add(.5).clamp(0,255).byte().permute(1,2,0).contiguous().cpu().numpy()
   return pix,dict(arm=arm,index=index,query_id=q['query_id'],stimulus_id=q['stimulus_id'],array_row=q['array_row'],EEG_row=None if arm=='uniform' else row,native_class_id=k,class_draw_seed=cs,token_seed=ts,
    EEG_logit_row_sha256=hashlib.sha256(logits.cpu().numpy().tobytes()).hexdigest(),native_class_table_SHA256=table_sha,native_condition_row_sha256=hashlib.sha256(table[k].cpu().numpy().tobytes()).hexdigest(),
    pixel_SHA256=hashlib.sha256(pix.tobytes()).hexdigest(),model_SHA256=model_sha[host],training_source_host=host,execution_host='old',cached_fixed_final_EEG_logits=True,
    generation=dict(one_class_draw=True,one_image=True,best_of_N=1,guidance=0.,top_k=0,top_p=0.,resolution=[256,256],scales=[1,2,3,4,5,6,8,10,13,16]),normal_inference_reads_query_image_label_or_gallery=False,labels_used_only_to_construct_experimental_EEG_permutations=(arm in ('same_donor','different_donor')))
  # Verify all400 original class decisions from saved probabilities on OLD;
  # one original image per arm is also replayed before generating new queries.
  phase='EXACT_OLD_HOST_CLASS_AND_PIXEL_MIGRATION_REPLAY';replay_paths={}
  if lo==80:
   for arm in ARMS:
    for i,r in enumerate(records80[arm]):
     host='old' if arm in ('coupled','same_donor','different_donor') else 'new';row=same[i] if arm=='same_donor' else different[i] if arm=='different_donor' else i
     logits=torch.zeros((1,1000),device='cuda') if arm=='uniform' else source[host][row:row+1]
     assert r['EEG_row']==(None if arm=='uniform' else row) and r['EEG_logit_row_sha256']==hashlib.sha256(logits.cpu().numpy().tobytes()).hexdigest()
     k=int(draw_native_class(logits,generator=torch.Generator(device='cuda').manual_seed(r['class_draw_seed']))[0]);assert k==r['native_class_id']
    pix,r=image(arm,0);expected=records80[arm][0]
    for key in ('native_class_id','class_draw_seed','token_seed','pixel_SHA256','native_condition_row_sha256'):assert r[key]==expected[key],(arm,key)
    p=out/f'engineering_original80_replay_{arm}.png';Image.fromarray(pix).save(p);assert ref(p)['sha256']==expected['image']['sha256'];replay_paths[arm]=p
  phase='FIXED_CONDITIONS_ONE_IMAGE_FOR_MISSING_QUERIES';records={};paths={}
  for arm in ARMS:
   directory=out/arm;directory.mkdir();records[arm]=[];paths[arm]=[]
   for index in range(lo,hi):
    guard();pix,r=image(arm,index);p=directory/f'{index:04d}.png';Image.fromarray(pix).save(p);r['image']=ref(p);records[arm].append(r);paths[arm].append(p)
   write(directory/'records.json',records[arm])
  assert digest(d.state_dict())==before and digest(vae.state_dict())==vq_before
  del native,vae,d,source,table;gc.collect();torch.cuda.empty_cache();guard();phase='FROZEN_DINO_AND_REPLAY_FEATURES'
  import image_evaluation as E
  assert sha(E.__file__)==s['image_evaluation']['sha256'] and E.DINO_SHA256==s['DINO_weight']['sha256'];check(s['DINO_weight']);model=E.dino_model();features={};errors={}
  for arm in ARMS:
   x=E.dino_features(paths[arm],model,batch_size=20,real_images=False).float().cpu().numpy();assert x.shape==(20,768) and np.isfinite(x).all();p=out/arm/'features.npy';np.save(p,x,allow_pickle=False);features[arm]=ref(p)
   if lo==80:
    # Use the original batch20 recipe for equality; each row is independent.
    xx=E.dino_features([replay_paths[arm]]*20,model,batch_size=20,real_images=False).float().cpu().numpy();errors[arm]=float(np.max(np.abs(xx[0]-features80[arm][0])));assert errors[arm]<2e-6
  C.verify_protected_files(a);guard()
  write(out/'completion.json',dict(status='COMPLETE_CATEGORICAL_OLD_ONLY_FIVE_ARM_IMAGE_CHUNK_v2',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),spec=ref(sp),range=[lo,hi],arms=ARMS,execution_host='old',source_bindings=s['source_bindings'],native_class_table_SHA256=table_sha,
   records={arm:ref(out/arm/'records.json') for arm in ARMS},features=features,DINO_weight=s['DINO_weight'],image_evaluation=s['image_evaluation'],native_VAR_VQ_unchanged=True,normal_input_EEG_only=True,effect_scores_read=False,target_images_read=0,new_research_images=100,
   original80_class_decisions_replayed=400 if lo==80 else 0,original80_pixel_replays=len(replay_paths),original80_feature_max_errors=errors,seconds=time.monotonic()-started,peak_GPU_allocated=torch.cuda.max_memory_allocated(),extra_CPU_peak=peak()-base))
 except BaseException as e:
  write(out/'failure.json',dict(status='CATEGORICAL_OLD_ONLY_IMAGE_FAILURE_PRESERVED_NO_RETRY',phase=phase,error=repr(e),traceback=traceback.format_exc(),range=[lo,hi],seconds=time.monotonic()-started));raise


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--job-name',required=True);p.add_argument('--lo',type=int,required=True);p.add_argument('--hi',type=int,required=True);q=p.parse_args();run(q.spec,q.lo,q.hi,q.job_name)
