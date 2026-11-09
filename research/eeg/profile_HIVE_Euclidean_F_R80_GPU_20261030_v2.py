"""All10 original parents replayed before a fixed loss_control R80 benchmark."""
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path


def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def ref(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))


def check(x):
 p=Path(x['path']);assert sha(p)==x['sha256']
 if 'bytes' in x:assert p.stat().st_size==x['bytes']
 return p


def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')


def memory():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024


def run(path,job):
 s=json.loads(Path(path).read_bytes());assert s['status']=='REGISTERED_FIXED_EUCLIDEAN_F_R80_MATCHED_BENCHMARK_AMENDED_FP32_v2' and s['host']=='old'
 for item in s['source_pins']:check(item)
 assert 'run_gpu_job_v2.py' in (Path('/proc')/str(os.getppid())/'cmdline').read_bytes().decode()
 assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==s['GPU_UUID']
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 R=Path(s['repository_root']);sys.path.insert(0,str(R));import brainhive_paired_study_contract_20261025_v1 as C
 C.setup_runtime(dict(runtime_manifest=s['runtime_manifest']))
 import torch
 import numpy as np
 from PIL import Image
 from brainhive_ordinary_geometry_control_20261030_v1 import EuclideanConditionControl
 torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 out=Path(s['output_root'])/'profile';out.mkdir(parents=True);started=time.monotonic();base=memory();phase='EXACT_RECOVERED_INPUT_AND_EUCLIDEAN_F'
 try:
  restored=json.loads(check(s['input_completion']).read_bytes());assert restored['queries']==restored['old_manifest_raw_condition_SHA_matches']==2000 and restored['all_old_raw_conditions_bit_exact']
  certificate=json.loads(check(s['numeric_certificate']).read_bytes());assert certificate['status']=='PASS_FINITE_CAPTURED_FP32_LAYER_CHECKS_ORIGINAL_FULL64_GUARD_STILL_FAILED' and not certificate['original_full64_guard_passed']
  certified=np.load(check(s['certified_FP32_trace']),allow_pickle=False)['output']
  bank=np.load(check(restored['bank']),allow_pickle=False);assert bank['conditions'].shape==(10,200,1024) and bank['gammas'].shape==bank['betas'].shape==(10,1024)
  state=torch.load(check(s['loss_control']),weights_only=True,map_location='cpu');assert state['arm']=='Euclidean_F' and state['geometry']=='UNBOUNDED_AFFINE_RESIDUAL' and state['updates']==24807 and state['seen_queries']==396900
  assert state['calibration']['selected_strength']==s['loss_control_strength'] and s['loss_control_strength'] in [0.,.25,.5,.75,1.]
  loss_control=EuclideanConditionControl().eval().requires_grad_(False);loss_control.load_state_dict(state['model'],strict=True);loss_control.radius=.25*s['loss_control_strength']
  all_c=torch.from_numpy(bank['conditions'].reshape(2000,1024).copy());all_w=torch.from_numpy(np.repeat(bank['gammas'],200,axis=0).copy());all_b=torch.from_numpy(np.repeat(bank['betas'],200,axis=0).copy())
  zero=EuclideanConditionControl().eval().requires_grad_(False)
  with torch.inference_mode():
   assert torch.equal(zero(all_c,all_w,all_b)[0],all_c)
   assert torch.equal(loss_control(all_c,all_w,all_b,enabled=False)[0],all_c)
  del zero,all_c,all_w,all_b
  rows=[json.loads(line) for line in check(s['baseline_inference']).read_text().splitlines()];assert len(rows)==2000
  keyed={r['query_id']:r for r in rows};assert len(keyed)==2000
  def budget():assert time.monotonic()-started<800 and memory()-base<10<<30 and torch.cuda.max_memory_allocated()<20<<30 and shutil.disk_usage(out).free>s['reserved_disk_bytes']
  phase='EXACT_META_WEIGHT_LOAD';from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
  from brainhive_component_generator_v1 import load_native
  from generate_BrainHIVE_neural_clean_fastscreen_20261027_v1 import sample_observed,save_image
  (out/'converted').mkdir();observer=ConvertedWeightAudit(torch,out/'converted');observer.install()
  try:pipe,adapter,projector,prompts,host=load_native(training=False)
  finally:observer.restore()
  loading=observer.complete();loss_control.cuda();budget();phase='ALL10_ORIGINAL_NATIVE_PIXEL_AND_EUCLIDEAN_F_PARITY'
  reports=[]
  for k,item in enumerate(restored['reports']):
   subject=item['subject'];reference=next(r for r in rows if r['subject']==subject and r['array_row']==0);assert ref(reference['generated_path'])['sha256']==reference['generated_sha256']
   c=torch.from_numpy(bank['conditions'][k,:1].copy()).cuda();w=torch.from_numpy(bank['gammas'][k:k+1].copy()).cuda();b=torch.from_numpy(bank['betas'][k:k+1].copy()).cuda()
   assert hashlib.sha256(c.cpu().numpy().tobytes()).hexdigest()==reference['condition_sha256']
   with torch.inference_mode():
    parent_tokens=projector(adapter(c.half()));native,proof=sample_observed(pipe,prompts,parent_tokens,reference['image_seed'],torch)
   with Image.open(reference['generated_path']) as im:expected=np.asarray(im.convert('RGB')).copy()
   assert np.array_equal(np.asarray(native),expected),(subject,'OriginalR80 parent pixel mismatch; do not reuse old baseline')
   parent_file=save_image(out/(subject+'_native_replay.webp'),native)
   zero_GPU=EuclideanConditionControl().eval().requires_grad_(False).cuda()
   with torch.inference_mode():
    zc=zero_GPU(c,w,b)[0];dc=loss_control(c,w,b,enabled=False)[0];assert torch.equal(zc,c) and torch.equal(dc,c)
    zero_image,zero_proof=sample_observed(pipe,prompts,projector(adapter(zc.half())),reference['image_seed'],torch)
    disabled_image,disabled_proof=sample_observed(pipe,prompts,projector(adapter(dc.half())),reference['image_seed'],torch)
   assert np.array_equal(np.asarray(zero_image),expected) and np.array_equal(np.asarray(disabled_image),expected)
   for observed in (zero_proof,disabled_proof):
    assert observed['initial_noise_latents_sha256']==proof['initial_noise_latents_sha256'] and observed['ancestral_noise_observations']==proof['ancestral_noise_observations']
   zero_file=save_image(out/(subject+'_zero_head_replay.webp'),zero_image);disabled_file=save_image(out/(subject+'_disabled_replay.webp'),disabled_image);del zero_GPU
   with torch.inference_mode():
    corrected,info=loss_control(c,w,b);tokens=projector(adapter(corrected.half()));image,loss_control_proof=sample_observed(pipe,prompts,tokens,reference['image_seed'],torch)
   assert proof['initial_noise_latents_sha256']==loss_control_proof['initial_noise_latents_sha256'] and proof['ancestral_noise_observations']==loss_control_proof['ancestral_noise_observations']
   cpu=EuclideanConditionControl().eval().requires_grad_(False);cpu.load_state_dict(state['model'],strict=True);cpu.radius=.25*s['loss_control_strength']
   with torch.inference_mode():cpu_corrected=cpu(c.cpu(),w.cpu(),b.cpu())[0]
   error=float((cpu_corrected-corrected.cpu()).abs().max());scaled_error=float(((cpu_corrected-corrected.cpu()).abs()/(1+cpu_corrected.abs())).max());assert np.array_equal(corrected.cpu().numpy().view(np.uint32),certified[k*200:k*200+1].view(np.uint32))
   u=(c-b)/w;v=(corrected-b)/w;norm_error=float((u.norm()-v.norm()).abs()/u.norm());assert torch.isfinite(corrected.half()).all()
   saved=save_image(out/(subject+'_loss_control_engineering.webp'),image)
   reports.append(dict(subject=subject,query_id=reference['query_id'],parent_reference=reference,parent_replay=parent_file,zero_head_replay=zero_file,disabled_replay=disabled_file,zero_proof=zero_proof,disabled_proof=disabled_proof,loss_control_engineering=saved,parent_proof=proof,loss_control_proof=loss_control_proof,CPU_GPU_correction_max_abs_error=error,relative_standardized_norm_change=norm_error,CPU_GPU_scaled_error=scaled_error,original_CPU_GPU_scaled_guard_passed=scaled_error<2e-6,certified_singleton_GPU_condition_bit_exact=True,
    original_condition_SHA256=reference['condition_sha256'],corrected_condition_SHA256=hashlib.sha256(corrected.cpu().numpy().tobytes()).hexdigest(),model_checkpoint=item['model'],loss_control_checkpoint=s['loss_control']))
   del cpu;budget()
  assert sha(s['loss_control']['path'])==s['loss_control']['sha256']
  write(out/'completion.json',dict(status='PASS_ALL10_EUCLIDEAN_F_NATIVE_PIXELS_AND_CAPTURED_FP32_CERTIFICATE_AMENDED_v2',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),spec=ref(path),reports=reports,loader=loading,host=host,
   all2000_zero_head_and_disable_exact=True,all10_parent_pixels_exact=True,all10_zero_and_disabled_pixels_exact=True,all10_loss_control_native_RNG_paired=True,loss_control_strength=s['loss_control_strength'],loss_control_radius=.25,geometry='UNBOUNDED_AFFINE_RESIDUAL',radius_parameter_is_residual_scale_only=True,model_updates=0,new_research_images=0,engineering_images=40,effect_scores_read=False,
   original_fullT_FP64_guard_passed=False,numeric_verification_amended_after_original_failure=True,all10_certified_singleton_conditions_bit_exact=True,
   seconds=time.monotonic()-started,extra_CPU_peak=memory()-base,peak_GPU_allocated=torch.cuda.max_memory_allocated()))
 except BaseException as error:
  write(out/'failure.json',dict(status='EUCLIDEAN_F_R80_PROFILE_FAILURE_PRESERVED_NO_AUTOMATIC_RETRY',phase=phase,error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-started));raise


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--job-name',required=True);q=p.parse_args();run(q.spec,q.job_name)
