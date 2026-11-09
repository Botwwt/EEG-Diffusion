"""Own-final-gated finite five-arm image lifecycle; no partial effect reads."""
import argparse,datetime,hashlib,json,subprocess,time,traceback
from pathlib import Path
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def run(path,stage):
 s=json.loads(Path(path).read_bytes());assert s['status']=='REGISTERED_NATIVE_CATEGORICAL_OWN_FINAL_IMAGES_v1' and stage in (80,400)
 root=Path(s['queue_root'])/f'stage{stage}';root.mkdir(parents=True);started=time.monotonic();events=[]
 try:
  for x in s['source_pins']:assert ref(Path(x['path']))==x
  fit=Path(s['fit_root']);fq=Path(s['fit_queue_root'])
  while not (fit/'completion.json').is_file() or not (fq/'completion.json').is_file():
   assert time.monotonic()-started<3600 and not (fq/'failure.json').is_file()
   identity=s['training_controller'];p=Path('/proc')/str(identity['controller_PID'])/'stat';assert p.is_file(),'Original fit handle missing, do not restart'
   f=p.read_text().rsplit(')',1)[1].split();assert int(f[19])==identity['start_ticks'] and f[0] not in ('Z','X');time.sleep(5)
  actual=json.loads((fq/'completion.json').read_bytes());assert actual['status']=='COMPLETE_NATIVE_CATEGORICAL_FULL_T_FIXED_FINAL_AND_CPU_AUDIT'
  value=dict(s['image_template'],fit_completion=ref(fit/'completion.json'),fit_CPU_audit=ref(fq/'independent_CPU_audit.json'))
  if stage==400:value['stage80_decision']=s['stage80_decision']
  sp=Path(s['code'])/f"native_categorical_images_{s['host']}_stage{stage}_spec_v1.json";write(sp,value)
  for lo in range(0 if stage==80 else 80,stage,20):
   assert time.monotonic()-started<7200 and not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
   wrapper='run_gpu_job_v2.py' if s['host']=='old' else 'run_BrainHIVE_two_host_GPU_job_20261028_v1.py';job=f"native_categorical_{s['host']}_images{lo:03d}_20261029_v1"
   cmd=['/root/miniconda3/bin/python','-B',str(Path(s['repository_root'])/wrapper),'--name',job,'--max-seconds','900','--','/root/miniconda3/bin/python','-B',str(Path(s['code'])/'generate_EEG_native_categorical_images_GPU_20261029_v1.py'),'--spec',str(sp),'--lo',str(lo),'--hi',str(lo+20),'--job-name',job]
   with (root/(job+'.stdout.log')).open('xb') as so,(root/(job+'.stderr.log')).open('xb') as se:p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=so,stderr=se,start_new_session=True,cwd=s['code']);rc=p.wait(timeout=950)
   events.append(dict(job=job,returncode=rc,range=[lo,lo+20]));assert rc==0,events[-1]
   chunk=Path(value['output_root'])/f'chunk_{lo:04d}_{lo+20:04d}/completion.json';c=json.loads(chunk.read_bytes());assert c['status']=='COMPLETE_NATIVE_CATEGORICAL_FIXED_FINAL_IMAGE_FEATURE_CHUNK' and c['range']==[lo,lo+20] and c['spec']==ref(sp) and not c['effect_scores_read']
  write(root/'completion.json',dict(status='COMPLETE_NATIVE_CATEGORICAL_ALL_OWN_STAGE_IMAGES_FEATURES',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=s['host'],stage=stage,arms=value['arms'],image_spec=ref(sp),fit_completion=value['fit_completion'],events=events,seconds=time.monotonic()-started,effect_scores_read=False))
 except BaseException as e:write(root/'failure.json',dict(status='NATIVE_CATEGORICAL_IMAGE_QUEUE_FAILURE_PRESERVED_NO_RETRY',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),error=repr(e),traceback=traceback.format_exc(),events=events,seconds=time.monotonic()-started));raise
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);ap.add_argument('--stage',type=int,required=True);v=ap.parse_args();run(v.spec,v.stage)
