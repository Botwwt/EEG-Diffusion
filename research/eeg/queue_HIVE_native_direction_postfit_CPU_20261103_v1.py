"""Finite dependent postfit queue; never starts before all30 source fits and owner exit."""
import argparse,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
import brainhive_native_direction_image_contract_20261103_v1 as I
def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);a=p.parse_args();s,train,e,old,base,G,H,C=I.setup(a.protocol);root=Path(s['output_root']);out=root/'queue';assert not out.exists();out.mkdir();started=time.monotonic();active=None;worker=None;jobpath=None;execution_started=None
 def stop(sig,frame):raise SystemExit('Finite postfit controller received signal '+str(sig))
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 def log(value):
  record=value|dict(UTC=G.datetime.datetime.now(G.datetime.timezone.utc).isoformat());C.atomic_json(out/'progress.json',record)
  with (out/'events.jsonl').open('a') as f:f.write(json.dumps(record)+'\n');f.flush()
  print(json.dumps(record),flush=True)
 def cpu(name,script,extra):
  nonlocal worker,active,jobpath
  assert time.monotonic()-execution_started<s['maximum_execution_seconds']-420;jobpath=None;active=name;cmd=[sys.executable,'-B',str(Path(a.protocol).parent/script),'--protocol',a.protocol]+extra;env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
  with (out/(name+'.stdout.log')).open('xb') as stdout,(out/(name+'.stderr.log')).open('xb') as stderr:
   worker=subprocess.Popen(cmd,cwd=Path(a.protocol).parent,env=env,stdout=stdout,stderr=stderr,start_new_session=True);G.write(out/(name+'_process.json'),dict(PID=worker.pid,command=cmd));log(dict(stage=name,CPU_PID=worker.pid));rc=worker.wait(timeout=s['CPU_stage_seconds'])
  assert rc==0,(name,rc);worker=None
 def gpu(name,mode,lo=0):
  nonlocal worker,active,jobpath
  seconds=s['profile_seconds'] if mode=='profile' else s['generation_segment_seconds'];assert time.monotonic()-execution_started<s['maximum_execution_seconds']-seconds-60
  active=name;job='HIVE_native_EEG_postfit_20261103_v'+str(s['runtime_revision'])+'_'+name;jobpath=Path(base['repository_root'])/'audit'/(job+'_job.json')
  cmd=[sys.executable,'-B',str(Path(base['repository_root'])/'run_gpu_job_v2.py'),'--name',job,'--max-seconds',str(seconds),'--',sys.executable,'-B',str(Path(a.protocol).parent/'run_HIVE_native_direction_final_images_GPU_20261103_v1.py'),'--protocol',a.protocol,'--mode',mode,'--lo',str(lo),'--job-name',job];env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
  with (out/(name+'.stdout.log')).open('xb') as stdout,(out/(name+'.stderr.log')).open('xb') as stderr:
   worker=subprocess.Popen(cmd,cwd=Path(a.protocol).parent,env=env,stdout=stdout,stderr=stderr,start_new_session=True);G.write(out/(name+'_process.json'),dict(PID=worker.pid,command=cmd));log(dict(stage=name,wrapper_PID=worker.pid));rc=worker.wait(timeout=seconds+60)
  record=C.read(jobpath);assert rc==0 and record['status']=='COMPLETED' and record['exit_code']==0 and record['wrapper_pid']==worker.pid;G.write(out/(name+'_closed.json'),dict(job=C.ref(jobpath),wrapper_returncode=rc));worker=None
 try:
  proof=C.read(root/'CPU_preflight/completion.json');assert proof['status']=='PASS_CPU_SOURCE_MATH_REAL10_PARENT_PROBES_AND_FIXED_GALLERY_POLICY_NOT_POSTFIT_GPU_RESULT' and proof['protocol']==C.ref(a.protocol)
  last=None
  while True:
   state=I.dependency(s,train,C)
   if state['status']=='READY_ALL30_FIXED_FINALS_AND_ACTUAL_OWNER_EXIT':break
   if state['status']!='WAIT_SAME_ORIGINAL_TRAINING_OWNER':
    G.write(out/'withdrawn.json',dict(status='WITHDRAWN_BEFORE_POSTFIT_GPU_SOURCE_TRAINING_NOT_SUCCESSFUL',source=state,protocol=C.ref(a.protocol),new_GPU_jobs=0));return
   assert time.time()<s['source_wait_deadline_epoch'],'Original finite source deadline exceeded; do not reset it'
   if state['status']!=last:log(dict(stage=state['status'],source_owner=s['training_owner'],source_state=state['owner_state'],new_GPU_jobs=0));last=state['status']
   time.sleep(30)
  execution_started=time.monotonic();before=I.resources(s,train,e,C,True,True);G.write(out/'actual_source_ready.json',dict(source=state,protocol=C.ref(a.protocol),resources=before,UTC=G.datetime.datetime.now(G.datetime.timezone.utc).isoformat()))
  cpu('export_conditions','export_HIVE_native_direction_final_Dcheck_CPU_20261103_v1.py',[]);assert C.read(root/'conditions/completion.json')['status']=='COMPLETE_ALL400_FOUR_ARM_FIXED_FINAL_CPU_EEG_CONDITIONS'
  gpu('profile','profile');assert C.read(root/'profile/completion.json')['status']=='PASS_NATIVE_PARITIES_FOUR_FINAL_CONDITIONS_AND_DINO_CACHE_REPLAY'
  gpu('images000_080','generate',0)
  cpu('audit80_inputs','audit_HIVE_native_direction_final_inputs_CPU_20261103_v1.py',['--stage','80']);assert C.read(root/'audit80/inputs.json')['queries']==80
  cpu('score80','score_HIVE_native_direction_final_images_CPU_20261103_v1.py',['--stage','80'])
  cpu('audit80_scores','audit_HIVE_native_direction_final_scores_CPU_20261103_v1.py',['--stage','80']);assert C.read(root/'audit80/scores.json')['status']=='PASS_INDEPENDENT_ALL7_METHODS_ALL8_METRICS_RANKS_PIXELS_AND1000_PAIRED_DRAWS'
  first=C.read(root/'score80/completion.json');stage=80
  if first['decision']['action']=='EXPAND400':
   for lo in (80,160,240,320):gpu(f'images{lo:03d}_{lo+80:03d}','generate',lo)
   cpu('audit400_inputs','audit_HIVE_native_direction_final_inputs_CPU_20261103_v1.py',['--stage','400'])
   cpu('score400','score_HIVE_native_direction_final_images_CPU_20261103_v1.py',['--stage','400'])
   cpu('audit400_scores','audit_HIVE_native_direction_final_scores_CPU_20261103_v1.py',['--stage','400']);assert C.read(root/'audit400/scores.json')['status']=='PASS_INDEPENDENT_ALL7_METHODS_ALL8_METRICS_RANKS_PIXELS_AND1000_PAIRED_DRAWS';stage=400
  else:assert first['decision']['action']=='STOP_EXPANSION_JOINT_HARM'
  final=C.read(root/f'score{stage}/completion.json');G.write(out/'completion.json',dict(status='COMPLETE_FIXED_NATIVE_EEG_POSTFIT_STAGE_AND_INDEPENDENT_INPUT_SCORE_AUDITS',protocol=C.ref(a.protocol),source_training=state['source_completion'],stage=stage,
   readout=C.ref(root/f'score{stage}/completion.json'),inputs_audit=C.ref(root/f'audit{stage}/inputs.json'),scores_audit=C.ref(root/f'audit{stage}/scores.json'),decision=final['decision'],
   new_research_images=stage*4,cached_baseline_images_reused=stage*3,engineering_parity_images=3,no_duplicate_query0_research_generation=True,
   seconds_including_dependency=time.monotonic()-started,postfit_seconds=time.monotonic()-execution_started,no_automatic_model_or_epoch_selection=True,NEW_contacted=False,independent_confirmation=False,SOTA=False))
  log(dict(stage='COMPLETE',queries=stage,decision=final['decision']))
 except BaseException as error:
  if worker is not None and worker.poll() is None:
   if jobpath is not None and jobpath.exists():
    record=C.read(jobpath);assert record['wrapper_pid']==worker.pid;child=record['pid']
    if Path('/proc',str(child)).exists():
     command=Path('/proc',str(child),'cmdline').read_bytes();assert b'run_HIVE_native_direction_final_images_GPU_20261103_v1.py' in command;os.killpg(child,signal.SIGTERM)
     try:worker.wait(timeout=35)
     except subprocess.TimeoutExpired:
      if Path('/proc',str(child)).exists():os.killpg(child,signal.SIGKILL)
      worker.wait(timeout=15)
   else:
    os.killpg(worker.pid,signal.SIGTERM)
    try:worker.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(worker.pid,signal.SIGKILL);worker.wait(timeout=10)
  G.write(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),active=active,seconds=time.monotonic()-started,no_automatic_retry=True,source_training_not_modified=True));raise
if __name__=='__main__':main()
