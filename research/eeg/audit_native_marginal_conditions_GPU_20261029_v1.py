"""Independent all400 CPU EEG forward and all800 latent CUDA seed replays.

Native condition distance is descriptive geometry, not a causal explanation or
an uncertainty-calibration result. No image, score or fitted weight is changed.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import argparse,datetime,hashlib,json,subprocess,sys,time,traceback
from pathlib import Path
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))
def check(x):p=Path(x['path']);assert ref(p)==x;return p
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def peak():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
def run(sp,job):
 s=json.loads(Path(sp).read_bytes());assert s['status']=='REGISTERED_NATIVE_CONDITION_AUDIT_AND_CLASS_LIKELIHOOD_ENGINEERING_v1'
 root=Path(s['output_root'])/'condition_audit';root.mkdir();started=time.monotonic();base=peak();phase='IMMUTABLE_FINALS_AND_WRAPPER'
 try:
  for x in s['source_pins']:check(x)
  host=s['host'];expected='run_gpu_job_v2.py' if host=='old' else 'run_BrainHIVE_two_host_GPU_job_20261028_v1.py'
  assert expected in (Path('/proc')/str(os.getppid())/'cmdline').read_bytes().decode().replace('\0',' ')
  assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==s['GPU_UUID']
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  receipt=Path(s['repository_root'])/'audit'/(job+'_job.json');r=json.loads(receipt.read_bytes());assert r['status']=='RUNNING' and r['pid']==os.getpid()
  assert float(r['max_seconds'] if host=='old' else r['maximum_seconds'])==900
  fs=json.loads(check(s['fit_spec']).read_bytes());sys.path.insert(0,fs['training_code']);import AR_anchored_training_contract_CPU_20261029_v1 as C
  a,_=C.load_admission(fs['training_admission']);C.validate_sources(a);C.verify_protected_files(a);sys.path.insert(0,a['vendor'])
  import numpy as np
  import torch
  from eeg_labram_pretrained_20261028_v1 import build_encoder
  from eeg_labram_interface_20261028_v1 import CacheNativeEEGTransform
  from EEG_native_conditional_variational_core_20261029_v1 import EEGGaussianPrior,reparameterize
  torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  def guard():
   assert time.monotonic()-started<800 and peak()-base<10<<30
   assert not torch.cuda.is_initialized() or torch.cuda.max_memory_allocated()<20<<30
  image_queue=json.loads(check(s['image_queue_spec']).read_bytes());image_spec=json.loads(check(s['image_spec']).read_bytes())
  done=json.loads(check(image_spec['fit_completion']).read_bytes());checkpoint=torch.load(check(done['final_model']),weights_only=True,map_location='cpu')
  assert checkpoint['epoch']==10 and checkpoint['updates']==2070
  model_state=checkpoint['model'];center=model_state['native_center'];logs0=model_state['native_logscale'];scale=logs0.exp()
  p=EEGGaussianPrior(build_encoder(a['author_source'],a['vendor'],seed=1729),center,scale).eval().requires_grad_(False);p.load_state_dict(model_state,strict=True)
  stats=json.loads(check(fs['full_EEG']['normalization']).read_bytes());assert stats['training_positions']==list(range(13230))
  raw=np.load(check(image_spec['Dcheck_mean63']),mmap_mode='r',allow_pickle=False);assert raw.shape==(400,63,250)
  transform=CacheNativeEEGTransform(tuple(stats['channel_names']),np.array(stats['center']),np.array(stats['scale']))
  from models.labram import get_input_chans
  channels=get_input_chans([x.upper() for x in stats['channel_names']]);means=[];logs=[];phase='INDEPENDENT_COMPLETE400_CPU_EEG_FORWARD'
  with torch.inference_mode():
   for begin in range(0,400,64):
    x=torch.from_numpy(transform(np.array(raw[begin:begin+64]),channel_names=tuple(stats['channel_names'])))
    m,l=p(x,input_chans=channels);means.append(m);logs.append(l);guard()
  means=torch.cat(means).numpy();logs=np.concatenate([x.numpy() for x in logs]);images=Path(image_spec['output_root']);mean_hashes=set();log_hashes=set();all_records={arm:[] for arm in image_spec['arms']};max_mean=0.;max_log=0.
  for lo in range(0,400,20):
   c=json.loads((images/f'chunk_{lo:04d}_{lo+20:04d}'/'completion.json').read_bytes());check(c['EEG_means']);check(c['EEG_logscales'])
   gm=np.load(c['EEG_means']['path'],allow_pickle=False);gl=np.load(c['EEG_logscales']['path'],allow_pickle=False)
   assert np.allclose(means,gm,atol=3e-5,rtol=3e-4) and np.allclose(logs,gl,atol=3e-5,rtol=3e-4)
   max_mean=max(max_mean,float(np.abs(means-gm).max()));max_log=max(max_log,float(np.abs(logs-gl).max()));mean_hashes.add(c['EEG_means']['sha256']);log_hashes.add(c['EEG_logscales']['sha256'])
   for arm in all_records:all_records[arm]+=json.loads(check(c['records'][arm]).read_bytes())
  assert len(mean_hashes)==len(log_hashes)==1
  phase='ALL800_CUDA_GAUSSIAN_SEED_REPLAY_AND_NATIVE_CONDITION_GEOMETRY';torch.cuda.reset_peak_memory_stats();cg=center.cuda()[None];lg=logs0.cuda()[None];samples={};errors={}
  for arm,records in all_records.items():
   assert len(records)==400;z_all=[];err=0.
   for i,r in enumerate(records):
    assert r['index']==i
    mu,log=(cg,lg) if arm=='marginal' else (torch.from_numpy(gm[r['EEG_row']:r['EEG_row']+1]).cuda(),torch.from_numpy(gl[r['EEG_row']:r['EEG_row']+1]).cuda())
    z=reparameterize(mu,log,generator=torch.Generator(device='cuda').manual_seed(r['latent_seed']))
    stored=np.load(check(r['latent']),allow_pickle=False);actual=z.cpu().numpy();assert np.array_equal(actual,stored)
    err=max(err,float(np.abs(actual-stored).max()));z_all.append(z);guard()
   errors[arm]=err;samples[arm]=torch.cat(z_all)
  public=torch.load(check(a['public_VAR']),weights_only=True,map_location='cpu',mmap=True);classes=public['class_emb.weight'][:-1].cuda();assert classes.shape==(1000,1024)
  native=(classes-cg)/lg.exp();native_pair=torch.cdist(native,native).square()/1024;native_pair.fill_diagonal_(torch.inf)
  native_nearest=native_pair.min(1).values
  geometry={arm:dict(nearest_native_class_standardized_MSE_median=float((torch.cdist((z-cg)/lg.exp(),native).square()/1024).min(1).values.median()),nearest_native_class_standardized_MSE_mean=float((torch.cdist((z-cg)/lg.exp(),native).square()/1024).min(1).values.mean())) for arm,z in samples.items()}
  u=(means-center.numpy())/scale.numpy();noise=float(np.exp(2*(logs-logs0.numpy())).mean());between=float(u.var(0).mean());bias=float(np.square(u.mean(0)).mean())
  C.verify_protected_files(a);guard()
  result=dict(status='PASS_ALL400_INDEPENDENT_CPU_EEG_AND_ALL800_EXACT_CUDA_LATENT_REPLAY',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=host,spec=ref(sp),model=done['final_model'],CPU_forward_queries=400,CPU_forward_scalar_parameters=400*2048,CPU_vs_original_GPU_mean_max_abs=max_mean,CPU_vs_original_GPU_logscale_max_abs=max_log,
   all20_chunks_prediction_arrays_bit_identical=True,latent_replays=800,latent_max_abs_errors=errors,native_class_leave_one_out_nearest_standardized_MSE_median=float(native_nearest.median()),sample_geometry=geometry,
   predicted_distribution_geometry=dict(between_query_normalized_mean_variance=between,global_normalized_mean_squared=bias,predicted_normalized_noise_variance=noise,interpretation='Descriptive native-coordinate moments, not calibrated EEG uncertainty or a causal explanation of image quality'),
   seconds=time.monotonic()-started,peak_GPU_allocated=torch.cuda.max_memory_allocated(),extra_CPU_peak=peak()-base,new_models_or_images=0)
  write(root/'completion.json',result)
 except BaseException as e:write(root/'failure.json',dict(status='CONDITION_REPLAY_FAILURE_PRESERVED',phase=phase,error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-started));raise
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);ap.add_argument('--job-name',required=True);v=ap.parse_args();run(v.spec,v.job_name)
