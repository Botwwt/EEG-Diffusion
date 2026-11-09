"""Finite actual64 profile, fullT fixed10 fit and independent CPU closure."""
import argparse,datetime,hashlib,json,subprocess,time,traceback
from pathlib import Path
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def run(path):
 s=json.loads(Path(path).read_bytes());root=Path(s['queue_root']);root.mkdir();started=time.monotonic();events=[]
 try:
  def stage(mode,sp,index):
   assert time.monotonic()-started<3300
   job=f"native_categorical_{s['host']}_{mode}{index}_20261029_v1";wrapper='run_gpu_job_v2.py' if s['host']=='old' else 'run_BrainHIVE_two_host_GPU_job_20261028_v1.py'
   cmd=['/root/miniconda3/bin/python','-B',str(Path(s['repository_root'])/wrapper),'--name',job,'--max-seconds','900','--','/root/miniconda3/bin/python','-B',str(Path(s['code'])/'train_EEG_native_categorical_20261029_v1.py'),'--spec',str(sp),'--job-name',job,'--mode',mode]
   with (root/(job+'.stdout.log')).open('xb') as so,(root/(job+'.stderr.log')).open('xb') as se:p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=so,stderr=se,start_new_session=True,cwd=s['code']);rc=p.wait(timeout=950)
   events.append(dict(job=job,returncode=rc));assert rc==0,events[-1]
   spec=json.loads(sp.read_bytes());done=Path(spec['output_root'])/(mode+'_'+job)/'completion.json';value=json.loads(done.read_bytes());assert value['spec']==ref(sp)
   return value,done
  profile=Path(s['profile_spec']['path']);assert ref(profile)==s['profile_spec'];value,done=stage('profile',profile,1)
  assert value['status']=='PASS_REAL_NATIVE_CATEGORICAL_64_QUERY_THREE_UPDATE_PROFILE' and value['backbone_gradient_nonzero'] and value['temporary_updates_discarded']
  fs=json.loads(profile.read_bytes());fs['output_root']=s['fit_root'];fs['profile_completion']=ref(done);fit_spec=Path(s['code'])/f"native_categorical_{s['host']}_fit_spec_v1.json";write(fit_spec,fs)
  for segment in range(1,4):
   value,done=stage('fit',fit_spec,segment)
   if value['status']=='COMPLETE_FIXED10_NATIVE_CATEGORICAL_FULL_T_FIT':break
   assert value['status']=='WHOLE_EPOCH_NATIVE_CATEGORICAL_SEGMENT_CLOSED'
  assert value['status']=='COMPLETE_FIXED10_NATIVE_CATEGORICAL_FULL_T_FIT','Finite budget exhausted with partial state retained'
  import torch
  assert ref(Path(value['checkpoint']['path']))==value['checkpoint'];state=torch.load(value['checkpoint']['path'],weights_only=False,map_location='cpu')
  assert state['epoch']==10 and state['updates']==2070 and len(state['events'])==2070 and len(state['coverage'])==10 and all(v['queries']==13230 for v in state['coverage'])
  assert all(torch.isfinite(v).all() for v in state['model'].values())
  steps=[int(v['step']) for v in state['optimizer']['state'].values()];assert steps and set(steps)=={2070}
  assert all(torch.isfinite(x).all() for v in state['optimizer']['state'].values() for x in v.values() if isinstance(x,torch.Tensor))
  final=torch.load(value['final_model']['path'],weights_only=True,map_location='cpu');assert ref(Path(value['final_model']['path']))==value['final_model'] and final['spec_sha256']==sha(fit_spec)
  assert set(final['model'])==set(state['model']) and all(torch.equal(v,final['model'][k]) for k,v in state['model'].items())
  assert state['CPU_RNG'].dtype==torch.uint8 and len(state['CUDA_RNG'])==1
  write(root/'independent_CPU_audit.json',dict(status='PASS_NATIVE_CATEGORICAL_FULL10_MODEL_ADAM_COVERAGE_AND_RNG_CPU',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),epochs=10,updates=2070,query_exposures=132300,optimizer_states=len(steps),model=value['final_model'],checkpoint=value['checkpoint'],new_images=0))
  write(root/'completion.json',dict(status='COMPLETE_NATIVE_CATEGORICAL_FULL_T_FIXED_FINAL_AND_CPU_AUDIT',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=s['host'],arm=s['arm'],events=events,fit_completion=ref(done),CPU_audit=ref(root/'independent_CPU_audit.json'),seconds=time.monotonic()-started,new_images=0,method_gain_or_SOTA_established=False))
 except BaseException as e:write(root/'failure.json',dict(status='NATIVE_CATEGORICAL_QUEUE_FAILURE_PRESERVED_NO_RETRY',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),error=repr(e),traceback=traceback.format_exc(),events=events,seconds=time.monotonic()-started));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();run(a.spec)
