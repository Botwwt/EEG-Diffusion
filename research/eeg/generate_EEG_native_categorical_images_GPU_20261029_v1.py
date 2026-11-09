"""Five predeclared interventions, one native class draw and one native image."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import argparse,datetime,gc,hashlib,json,shutil,subprocess,sys,time,traceback
from pathlib import Path
from generate_EEG_native_marginal_images_GPU_20261029_v2 import sha,ref,check,write,peak,seed,digest

def run(sp,lo,hi,job):
 s=json.loads(Path(sp).read_bytes());assert s['status']=='REGISTERED_NATIVE_CATEGORICAL_FIXED_FINAL_FIVE_ARM_IMAGES_v1' and 0<=lo<hi<=400 and lo%20==0 and hi==lo+20
 for x in s['source_pins']:check(x)
 if lo>=80:assert json.loads(check(s['stage80_decision']).read_bytes())['decision']=='EXPAND400'
 done=json.loads(check(s['fit_completion']).read_bytes());cpu=json.loads(check(s['fit_CPU_audit']).read_bytes());assert done['status']=='COMPLETE_FIXED10_NATIVE_CATEGORICAL_FULL_T_FIT' and done['epochs']==10 and done['updates']==2070
 assert cpu['status']=='PASS_NATIVE_CATEGORICAL_FULL10_MODEL_ADAM_COVERAGE_AND_RNG_CPU' and cpu['model']==done['final_model']
 cat=json.loads(check(done['spec']).read_bytes());assert cat['arm']==s['fitted_arm'] and cat['inference']['top_k']==0
 root=Path(s['output_root']);root.mkdir(exist_ok=True);out=root/f'chunk_{lo:04d}_{hi:04d}';out.mkdir();started=time.monotonic();base=peak();phase='EXCLUSIVE_WRAPPER_AND_RAW_EEG'
 try:
  expected='run_gpu_job_v2.py' if s['host']=='old' else 'run_BrainHIVE_two_host_GPU_job_20261028_v1.py';assert expected in (Path('/proc')/str(os.getppid())/'cmdline').read_bytes().decode().replace('\0',' ')
  assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==s['GPU_UUID'] and not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  rr=json.loads((Path(s['repository_root'])/'audit'/(job+'_job.json')).read_bytes());assert rr['status']=='RUNNING' and rr['pid']==os.getpid() and float(rr['max_seconds'] if s['host']=='old' else rr['maximum_seconds'])==900
  base_spec=json.loads(check(cat['base_EEG_spec']).read_bytes());sys.path.insert(0,base_spec['training_code']);import AR_anchored_training_contract_CPU_20261029_v1 as C
  a,_=C.load_admission(base_spec['training_admission']);C.validate_sources(a);C.verify_protected_files(a);sys.path.insert(0,a['vendor'])
  import numpy as np
  import torch
  from PIL import Image
  from EEG_generative_backend_loader_20261029_v1 import build_and_load
  from eeg_labram_pretrained_20261028_v1 import build_encoder
  from eeg_labram_interface_20261028_v1 import CacheNativeEEGTransform
  from EEG_native_conditional_variational_core_20261029_v1 import NativeConditionDecoder
  from EEG_native_categorical_core_20261029_v1 import EEGNativeClassPrior,draw_native_class
  torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  panel=json.loads(check(s['panel']).read_bytes());queries=panel['queries'];physical=json.loads(check(s['physical_queries']).read_bytes());assert len(queries)==len(physical['Dcheck'])==400
  for first,second in zip(queries,physical['Dcheck']):assert all(first[k]==second[k] for k in ('stimulus_id','concept_id','array_row'))
  groups={}
  for i,q in enumerate(queries):groups.setdefault(q['concept_id'],[]).append(i)
  keys=sorted(groups);assert len(keys)==40 and {len(g) for g in groups.values()}=={10}
  same={i:groups[k][(j+1)%10] for k in keys for j,i in enumerate(groups[k])};different={i:groups[keys[(n+1)%40]][j] for n,k in enumerate(keys) for j,i in enumerate(groups[k])}
  assert sorted(same.values())==sorted(different.values())==list(range(400))
  arms=['coupled','same_donor','different_donor'] if s['host']=='old' else ['visual','uniform'];assert s['arms']==arms
  raw=np.load(check(s['Dcheck_mean63']),mmap_mode='r',allow_pickle=False);stats=json.loads(check(base_spec['full_EEG']['normalization']).read_bytes());assert raw.shape==(400,63,250) and stats['training_positions']==list(range(13230))
  transform=CacheNativeEEGTransform(tuple(stats['channel_names']),np.array(stats['center']),np.array(stats['scale']));x=transform(np.array(raw),channel_names=tuple(stats['channel_names']))
  encoder=build_encoder(a['author_source'],a['vendor'],seed=1729);p=EEGNativeClassPrior(encoder);state=torch.load(check(done['final_model']),weights_only=True,map_location='cpu');assert state['epoch']==10 and state['updates']==2070 and state['spec_sha256']==done['spec']['sha256'] and state['arm']==s['fitted_arm'];p.load_state_dict(state['model'],strict=True);del state;p.cuda().eval().requires_grad_(False)
  from models.labram import get_input_chans
  channels=get_input_chans([k.upper() for k in stats['channel_names']])
  with torch.inference_mode():EEG_logits=torch.cat([p(torch.from_numpy(x[k:k+64]).cuda(),input_chans=channels) for k in range(0,400,64)])
  assert EEG_logits.shape==(400,1000) and torch.isfinite(EEG_logits).all();np.save(out/'EEG_logits400.npy',EEG_logits.cpu().numpy(),allow_pickle=False);del p;gc.collect();torch.cuda.empty_cache()
  phase='UNCHANGED_NATIVE_GENERATOR_AND_CLASS_TABLE';torch.cuda.reset_peak_memory_stats();vae,d,backend=build_and_load(a['author_source'],a['vendor'],check(a['public_VQ']),check(a['public_VAR']),seed=1729)
  public=torch.load(check(a['public_VAR']),weights_only=True,map_location='cpu',mmap=True);classes=public['class_emb.weight'].clone();del public
  before=digest({k:v for k,v in d.state_dict().items() if not k.startswith('embed_proj.')});vq_before=digest(vae.state_dict());native=NativeConditionDecoder(d,classes).cuda().eval();vae.cuda().eval().requires_grad_(False);d.rng=torch.Generator(device='cuda')
  native_table=classes[:1000].cuda();table_sha=hashlib.sha256(classes.numpy().tobytes()).hexdigest()
  def guard():
   assert time.monotonic()-started<800 and peak()-base<10<<30 and torch.cuda.max_memory_allocated()<20<<30 and shutil.disk_usage(out).free>2<<30
  records={};paths={};phase='EEG_ONLY_ONE_CLASS_AND_ONE_IMAGE_UNTRUNCATED_NATIVE_LAW'
  for arm in arms:
   directory=out/arm;directory.mkdir();r=[];pp=[]
   for index in range(lo,hi):
    guard();q=queries[index];source=same[index] if arm=='same_donor' else different[index] if arm=='different_donor' else index
    logits=torch.zeros((1,1000),device='cuda') if arm=='uniform' else EEG_logits[source:source+1];cs=seed(q['stimulus_id'],'latent');ts=seed(q['stimulus_id'],'token')
    k=int(draw_native_class(logits,generator=torch.Generator(device='cuda').manual_seed(cs))[0]);assert 0<=k<1000
    with torch.inference_mode():image=native.generate(native_table[k:k+1],seed=ts,guidance=0.,top_k=0)
    assert image.shape==(1,3,256,256) and torch.isfinite(image).all();pix=image[0].mul(255).add(.5).clamp(0,255).byte().permute(1,2,0).contiguous().cpu().numpy()
    if index==0:
     with torch.inference_mode():replay=native.generate(native_table[k:k+1],seed=ts,guidance=0.,top_k=0)
     rp=replay[0].mul(255).add(.5).clamp(0,255).byte().permute(1,2,0).cpu().numpy();assert np.array_equal(pix,rp);del replay,rp
    image_path=directory/f'{index:04d}.png';Image.fromarray(pix,mode='RGB').save(image_path)
    record=dict(arm=arm,index=index,query_id=q['query_id'],stimulus_id=q['stimulus_id'],array_row=q['array_row'],EEG_row=None if arm=='uniform' else source,native_class_id=k,class_draw_seed=cs,token_seed=ts,
     EEG_logit_row_sha256=hashlib.sha256(logits.cpu().numpy().tobytes()).hexdigest(),native_class_table_SHA256=table_sha,native_condition_row_sha256=hashlib.sha256(native_table[k].cpu().numpy().tobytes()).hexdigest(),
     image=ref(image_path),pixel_SHA256=hashlib.sha256(pix.tobytes()).hexdigest(),model_SHA256=done['final_model']['sha256'],generation=dict(one_class_draw=True,one_image=True,best_of_N=1,guidance=0.,top_k=0,top_p=0.,resolution=[256,256],scales=[1,2,3,4,5,6,8,10,13,16]),
     normal_inference_reads_query_image_label_or_gallery=False,labels_used_only_to_construct_experimental_EEG_permutations=(arm in ('same_donor','different_donor')))
    r.append(record);pp.append(image_path);del image
   write(directory/'records.json',r);records[arm]=r;paths[arm]=pp
  assert digest(d.state_dict())==before and digest(vae.state_dict())==vq_before;del native,vae,d,EEG_logits,native_table;gc.collect();torch.cuda.empty_cache();guard();phase='FROZEN_DINO_FEATURES'
  import image_evaluation as E
  assert sha(Path(E.__file__))==s['image_evaluation']['sha256'];check(s['DINO_weight']);assert E.DINO_SHA256==s['DINO_weight']['sha256'];model=E.dino_model();features={}
  for arm in arms:
   for r in records[arm]:
    with Image.open(check(r['image'])) as im:pix=np.array(im.convert('RGB'),copy=True);assert im.size==(256,256)
    assert hashlib.sha256(pix.tobytes()).hexdigest()==r['pixel_SHA256']
   xx=E.dino_features(paths[arm],model,batch_size=20,real_images=False).float().cpu().numpy();assert xx.shape==(20,768) and np.isfinite(xx).all();fp=out/arm/'features.npy';np.save(fp,xx,allow_pickle=False);features[arm]=ref(fp)
  C.verify_protected_files(a);guard();write(out/'completion.json',dict(status='COMPLETE_NATIVE_CATEGORICAL_FIXED_FINAL_IMAGE_FEATURE_CHUNK',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),spec=ref(sp),range=[lo,hi],arms=arms,fit_completion=s['fit_completion'],fit_CPU_audit=s['fit_CPU_audit'],EEG_logits=ref(out/'EEG_logits400.npy'),native_class_table_SHA256=table_sha,
   records={arm:ref(out/arm/'records.json') for arm in arms},features=features,DINO_weight=s['DINO_weight'],image_evaluation=s['image_evaluation'],native_VAR_VQ_unchanged=True,normal_input_EEG_only=True,effect_scores_read=False,target_images_read=0,new_research_images=20*len(arms),engineering_replays=len(arms) if lo==0 else 0,seconds=time.monotonic()-started,peak_GPU_allocated=torch.cuda.max_memory_allocated(),extra_CPU_peak=peak()-base))
 except BaseException as e:write(out/'failure.json',dict(status='NATIVE_CATEGORICAL_IMAGE_FAILURE_PRESERVED_NO_RETRY',phase=phase,error=repr(e),traceback=traceback.format_exc(),range=[lo,hi],seconds=time.monotonic()-started));raise
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);ap.add_argument('--job-name',required=True);ap.add_argument('--lo',type=int,required=True);ap.add_argument('--hi',type=int,required=True);v=ap.parse_args();run(v.spec,v.lo,v.hi,v.job_name)
