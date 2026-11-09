"""Two finite evidence jobs per host; exceptions stop, never retry a GPU job."""
import argparse,datetime,hashlib,json,subprocess,time,traceback
from pathlib import Path
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def run(path):
 s=json.loads(Path(path).read_bytes());root=Path(s['output_root']);root.mkdir();started=time.monotonic();events=[]
 try:
  for x in s['source_pins']:assert Path(x['path']).stat().st_size==x['bytes'] and sha(x['path'])==x['sha256']
  for task,worker in [('condition_audit','audit_native_marginal_conditions_GPU_20261029_v1.py'),('native_class_bank','build_native_class_coarse_likelihood_bank_GPU_20261029_v1.py')]:
   assert time.monotonic()-started<1900
   wrapper='run_gpu_job_v2.py' if s['host']=='old' else 'run_BrainHIVE_two_host_GPU_job_20261028_v1.py';job=f"native_support_{s['host']}_{task}_20261029_v1"
   cmd=['/root/miniconda3/bin/python','-B',str(Path(s['repository_root'])/wrapper),'--name',job,'--max-seconds','900','--','/root/miniconda3/bin/python','-B',str(Path(s['code'])/worker),'--spec',str(path),'--job-name',job]
   with (root/(task+'.stdout.log')).open('xb') as so,(root/(task+'.stderr.log')).open('xb') as se:
    p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=so,stderr=se,start_new_session=True,cwd=s['code']);rc=p.wait(timeout=950)
   events.append(dict(task=task,returncode=rc,job=job));assert rc==0,events[-1]
   expected=root/('condition_audit' if task=='condition_audit' else 'full_T' if s['host']=='old' else 'independent_T64')/'completion.json'
   done=json.loads(expected.read_bytes());assert done['status'] in ('PASS_ALL400_INDEPENDENT_CPU_EEG_AND_ALL800_EXACT_CUDA_LATENT_REPLAY','COMPLETE_FIXED_NATIVE_CLASS_COARSE_LIKELIHOOD_BANK_NO_EEG_FIT')
  write(root/'queue_completion.json',dict(status='NATIVE_CONDITION_REPLAY_AND_FIXED_LIKELIHOOD_ENGINEERING_CLOSED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=s['host'],events=events,seconds=time.monotonic()-started,new_EEG_fits=0,new_images=0))
 except BaseException as e:write(root/'queue_failure.json',dict(status='NATIVE_SUPPORT_QUEUE_FAILURE_PRESERVED_NO_RETRY',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),error=repr(e),traceback=traceback.format_exc(),events=events,seconds=time.monotonic()-started));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();run(a.spec)
