"""Fixed loss_control, genuine native R80, one image/query; no target access."""
import argparse,gc,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
from profile_HIVE_Euclidean_F_R80_GPU_20261030_v2 import sha,ref,check,write,memory


def digest(module):
 h=hashlib.sha256()
 for name,value in module.state_dict().items():
  x=value.detach().cpu().contiguous();h.update(name.encode());h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode());h.update(x.numpy().tobytes())
 return h.hexdigest()


def run(execution_path,lo,hi,job):
 execution=json.loads(Path(execution_path).read_bytes());s=json.loads(check(execution['protocol']).read_bytes())
 assert execution['status']=='BOUND_FIXED_EUCLIDEAN_F_R80_FULL_LIFECYCLE_AMENDED_FP32_v2' and execution['host']=='old'
 assert 0<=lo<hi<=2000 and lo%200==0 and hi==lo+200
 for r in s['source_pins']+execution['source_pins']:check(r)
 profile=json.loads(check(execution['profile_completion']).read_bytes());assert profile['status']=='PASS_ALL10_EUCLIDEAN_F_NATIVE_PIXELS_AND_CAPTURED_FP32_CERTIFICATE_AMENDED_v2' and profile['all10_parent_pixels_exact']
 assert 'run_gpu_job_v2.py' in (Path('/proc')/str(os.getppid())/'cmdline').read_bytes().decode()
 assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==s['GPU_UUID']
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 sys.path.insert(0,s['repository_root']);import brainhive_paired_study_contract_20261025_v1 as C
 C.setup_runtime(dict(runtime_manifest=s['runtime_manifest']))
 import torch,numpy as np
 from PIL import Image
 from brainhive_component_generator_v1 import load_native
 from brainhive_ordinary_geometry_control_20261030_v1 import EuclideanConditionControl
 from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
 from generate_BrainHIVE_neural_clean_fastscreen_20261027_v1 import sample_observed,save_image
 torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 root=Path(s['output_root']);out=root/'loss_control';out.mkdir(exist_ok=True);attempt=root/'generation_attempts'/f'{lo:04d}_{hi:04d}';attempt.mkdir(parents=True);started=time.monotonic();base=memory();phase='FIXED_COMPONENT_AND_RECOVERED_INPUTS'
 try:
  done=json.loads(check(s['input_completion']).read_bytes());bank=np.load(check(done['bank']),allow_pickle=False)
  certified=np.load(check(s['certified_FP32_trace']),allow_pickle=False)['output'];assert certified.shape==(2000,1024)
  baseline=[json.loads(x) for x in check(s['baseline_inference']).read_text().splitlines()];assert len(baseline)==2000
  expected=[(r['subject'],r['array_row']) for r in baseline];assert expected==[(f'sub-{n:02d}',i) for n in range(1,11) for i in range(200)]
  state=torch.load(check(s['loss_control']),weights_only=True,map_location='cpu');assert state['arm']=='Euclidean_F' and state['geometry']=='UNBOUNDED_AFFINE_RESIDUAL' and state['updates']==24807 and state['seen_queries']==396900
  assert state['calibration']['selected_strength']==s['loss_control_strength'] and s['loss_control_strength'] in [0.,.25,.5,.75,1.]
  loss_control=EuclideanConditionControl().eval().requires_grad_(False).cuda();loss_control.load_state_dict(state['model'],strict=True);loss_control.radius=.25*s['loss_control_strength']
  (attempt/'converted').mkdir();observer=ConvertedWeightAudit(torch,attempt/'converted');observer.install();phase='EXACT_META_NATIVE_LOAD'
  try:pipe,A,P,prompts,host=load_native(training=False)
  finally:observer.restore()
  loader=observer.complete();models={'UNet':pipe.unet,'VAE':pipe.vae,'A':A,'P':P};frozen={k:digest(v) for k,v in models.items()}
  def budget():assert time.monotonic()-started<800 and memory()-base<10<<30 and torch.cuda.max_memory_allocated()<20<<30 and shutil.disk_usage(out).free>2<<30
  def inputs(index):
   n,i=divmod(index,200);r=baseline[index];c=torch.from_numpy(bank['conditions'][n,i:i+1].copy()).cuda();w=torch.from_numpy(bank['gammas'][n:n+1].copy()).cuda();b=torch.from_numpy(bank['betas'][n:n+1].copy()).cuda()
   assert hashlib.sha256(c.cpu().numpy().tobytes()).hexdigest()==r['condition_sha256'];return r,c,w,b
  phase='ORIGINAL_SUBJECT_NATIVE_PIXEL_REPLAY';reference,c,w,b=inputs(lo)
  with torch.inference_mode():image,proof=sample_observed(pipe,prompts,P(A(c.half())),reference['image_seed'],torch)
  assert sha(reference['generated_path'])==reference['generated_sha256']
  with Image.open(reference['generated_path']) as im:original=np.asarray(im.convert('RGB')).copy()
  assert np.array_equal(original,np.asarray(image));replay=save_image(attempt/'native_original_replay.webp',image)
  write(attempt/'native_replay.json',dict(reference=reference,replay=replay,proof=proof,pixel_exact=True))
  (out/'records').mkdir(exist_ok=True);(out/'pixels').mkdir(exist_ok=True);corrected=[];records=[];phase='FULL_FIXED_R80_ONE_IMAGE_PER_QUERY'
  for index in range(lo,hi):
   budget();r,c,w,b=inputs(index);record_path=out/'records'/f'{index:04d}.json';image_path=out/'pixels'/f'{index:04d}.webp';assert not record_path.exists() and not image_path.exists()
   with torch.inference_mode():
    cc,info=loss_control(c,w,b);assert np.array_equal(cc.cpu().numpy().view(np.uint32),certified[index:index+1].view(np.uint32));assert torch.isfinite(cc.half()).all();tokens=P(A(cc.half()));image,observed=sample_observed(pipe,prompts,tokens,r['image_seed'],torch)
   saved=save_image(image_path,image);corrected.append(cc.cpu().numpy()[0])
   row=dict(query_id=r['query_id'],subject=r['subject'],stimulus_id=r['stimulus_id'],array_row=r['array_row'],global_query_index=index,repeat_protocol='R80',intervention='correct',EEG_source_stimulus_id=r['stimulus_id'],method='BrainHIVE_'+s['control_arm']+'_R80_T1323_seed1729',control_arm=s['control_arm'],image_seed=r['image_seed'],
    original_condition_sha256=r['condition_sha256'],condition_sha256=hashlib.sha256(cc.cpu().numpy().tobytes()).hexdigest(),actual_A_input_fp16_sha256=hashlib.sha256(cc.half().cpu().numpy().tobytes()).hexdigest(),actual_tokens_sha256=hashlib.sha256(tokens.cpu().numpy().tobytes()).hexdigest(),
    parent_checkpoint=done['reports'][index//200]['model'],loss_control_checkpoint=s['loss_control'],loss_control_strength=s['loss_control_strength'],loss_control_radius=.25,geometry='UNBOUNDED_AFFINE_RESIDUAL',radius_parameter_is_residual_scale_only=True,component_correction_RMS=float((cc-c).square().mean().sqrt()),gamma_sha256=hashlib.sha256(w.cpu().numpy().tobytes()).hexdigest(),beta_sha256=hashlib.sha256(b.cpu().numpy().tobytes()).hexdigest(),
    original_fullT_FP64_guard_passed=False,amended_numeric_verification=True,certified_singleton_condition_bit_exact=True,normal_inference_target_access=False,one_image_per_query=True,source_execution=ref(execution_path),**saved,**observed)
   write(record_path,row);records.append(row)
   if (index-lo+1)%25==0:print(json.dumps(dict(stage='FIXED_EUCLIDEAN_F_R80_GENERATION',lo=lo,saved=index-lo+1,queries=200,seconds=time.monotonic()-started)),flush=True)
  cp=attempt/'actual_corrected_conditions.npy';np.save(cp,np.array(corrected,np.float32),allow_pickle=False)
  after={k:digest(v) for k,v in models.items()};assert after==frozen
  assert all(torch.equal(loss_control.state_dict()[k].cpu(),v) for k,v in state['model'].items());assert sha(s['loss_control']['path'])==s['loss_control']['sha256'];budget()
  write(attempt/'completion.json',dict(status='COMPLETE_FIXED_EUCLIDEAN_F_R80_200_IMAGE_SEGMENT',execution=ref(execution_path),range=[lo,hi],queries=200,new_research_images=200,engineering_replays=1,corrected_conditions=ref(cp),records=[ref(out/'records'/f'{i:04d}.json') for i in range(lo,hi)],
   all_generator_and_loss_control_weights_exact=True,generator_state_hashes=frozen,loader=loader,host=host,effect_scores_read=False,seconds=time.monotonic()-started,extra_CPU_peak=memory()-base,peak_GPU_allocated=torch.cuda.max_memory_allocated()))
 except BaseException as error:
  write(attempt/'failure.json',dict(status='EUCLIDEAN_F_R80_GENERATION_FAILURE_PRESERVED_NO_AUTOMATIC_RETRY',phase=phase,error=repr(error),traceback=traceback.format_exc(),range=[lo,hi],seconds=time.monotonic()-started));raise


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--execution',required=True);p.add_argument('--lo',type=int,required=True);p.add_argument('--hi',type=int,required=True);p.add_argument('--job-name',required=True);q=p.parse_args();run(q.execution,q.lo,q.hi,q.job_name)
