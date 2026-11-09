"""Conditional full lifecycle; never restarts the upstream bank or failed tasks."""
import argparse,datetime,importlib.util,os,signal,subprocess,sys,time,traceback
from pathlib import Path
import EEG_complete_native_study_contract_CPU_20261102_v1 as C

def tasks_before_admission(q):
 return [('inputs','prepare_EEG_complete_native_inputs_CPU_20261102_v1.py',False,[],q['CPU_input_seconds'])]+[(a+'_profile','train_EEG_complete_native_study_GPU_20261102_v1.py',True,['--arm',a,'--phase','profile'],900) for a in C.ARMS]+[
  ('profiles_CPU','audit_EEG_complete_native_study_CPU_20261102_v1.py',False,['--mode','profiles'],q['CPU_profile_seconds']),
  ('generation_profile','generate_EEG_complete_native_study_GPU_20261102_v1.py',True,['--mode','profile'],900),
  ('generation_profile_CPU','audit_EEG_complete_native_predictions_images_CPU_20261102_v1.py',False,['--mode','profile'],q['CPU_artifact_seconds'])]

def tasks_for_fits(q):
 result=[]
 for arm in C.ARMS:
  for epoch in (5,10):
   result.append((arm+'_fit'+str(epoch),'train_EEG_complete_native_study_GPU_20261102_v1.py',True,['--arm',arm,'--phase','fit'+str(epoch)],900))
   result.append((arm+'_fit'+str(epoch)+'_CPU','audit_EEG_complete_native_study_CPU_20261102_v1.py',False,['--mode','fit'+str(epoch),'--arm',arm],q['CPU_fit_audit_seconds']))
 return result+[
  ('predict_final400','generate_EEG_complete_native_study_GPU_20261102_v1.py',True,['--mode','predict'],900),
  ('predictions_CPU','audit_EEG_complete_native_predictions_images_CPU_20261102_v1.py',False,['--mode','predictions'],q['CPU_prediction_seconds'])]

def tasks_for_images(q,stage):
 assert stage in (80,400)
 return [(f'images{stage}_{part:02d}','generate_EEG_complete_native_study_GPU_20261102_v1.py',True,['--mode','images','--stage',str(stage),'--part',str(part)],900) for part in range(4 if stage==80 else 16)]+[
  (f'artifacts{stage}_CPU','audit_EEG_complete_native_predictions_images_CPU_20261102_v1.py',False,['--mode',f'stage{stage}'],q['CPU_artifact_seconds']),
  (f'score{stage}_CPU','score_EEG_complete_native_study_CPU_20261102_v1.py',False,['--stage',str(stage)],q['CPU_score_seconds'])]

def run(protocol):
 q=C.setup(protocol,need_bank=False);assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 assert not (C.OUT/'queue_started.json').exists();started=time.monotonic();now=datetime.datetime.now(datetime.timezone.utc)
 C.write(C.OUT/'queue_started.json',dict(PID=os.getpid(),UTC=now.isoformat(),protocol=C.ref(protocol),formal_admission=False,waiting_on_exact_original_bank=[525462,119859959]))
 owner=pinned=None;owned=None;events=[];phase='WAIT_SAME_FULL_BANK';bank_deadline=datetime.datetime.fromisoformat(q['bank_started_UTC'])+datetime.timedelta(seconds=q['wait_for_bank_seconds'])
 try:
  while True:
   state=C.bank_state();C.write(C.OUT/'progress.json',dict(phase=phase,bank=state,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()),update=True)
   if state['status']=='READY_FULL_BANK_WHOLE_SUCCESS_EXACT_EXIT':break
   assert state['status']=='WAIT_SAME_BANK_LIVE',state
   assert datetime.datetime.now(datetime.timezone.utc)<bank_deadline and time.monotonic()-started<q['maximum_queue_seconds']
   C.resources(q,waiting=True);time.sleep(30)
  q=C.setup(protocol);C.resources(q,starting=True);_,bq=C.bank_contract();d,n,prev,_=C.bank_contract()[0].dependency(bq);m=n.mean_contract()
  pin=m.OWNED;spec=importlib.util.spec_from_file_location('owned_complete_EEG_study',m.checked(pin));owned=importlib.util.module_from_spec(spec);spec.loader.exec_module(owned)
  repo=Path(q['repository_root']);sys.path.insert(0,str(Path(m.PARENT['path']).parent))
  from brainhive_gpu_exit_contract_CPU_20261101_v1 import verify
  def task(item):
   nonlocal owner,pinned,phase
   label,entry,gpu,args,limit=item;phase=label;C.resources(q,starting=gpu)
   assert time.monotonic()-started<q['maximum_queue_seconds'];name='VAR_complete_EEG_20261102_v1_'+label
   command=[sys.executable,'-B',str(C.CODE/entry),'--protocol',str(protocol)]+args
   if gpu:
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    command=[sys.executable,'-B',str(repo/'run_gpu_job_v2.py'),'--name',name,'--max-seconds',str(limit),'--']+command+['--job-name',name]
   env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0' if gpu else '',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
   tick=time.monotonic();logs=C.OUT/'queue';logs.mkdir(exist_ok=True)
   with (logs/(label+'.stdout.log')).open('xb') as out,(logs/(label+'.stderr.log')).open('xb') as err:
    owner=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
    pinned=owned.capture_wrapper_child(owner,repo/'audit'/(name+'_job.json')) if gpu else None
    identity=C.process(owner.pid,-1);assert 'start_ticks' in identity
    C.write(logs/(label+'_identity.json'),dict(PID=owner.pid,start_ticks=identity['start_ticks'],GPU=gpu,job_name=name))
    C.write(C.OUT/'progress.json',dict(phase=label,completed_tasks=len(events),active_PID=owner.pid,active_start_ticks=identity['start_ticks'],GPU=gpu,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()),update=True)
    while owner.poll() is None:
     assert time.monotonic()-tick<limit+(65 if gpu else 0) and time.monotonic()-started<q['maximum_queue_seconds'];time.sleep(2)
    rc=owner.wait()
   event=dict(label=label,exit_code=rc,seconds=time.monotonic()-tick,GPU=gpu)
   if gpu:event.update(verify(rc,C.read(repo/'audit'/(name+'_job.json')),owner.pid));event['GPU_receipt']=C.ref(repo/'audit'/(name+'_job.json'))
   C.write(logs/(label+'_event.json'),event);events.append(event);owner=pinned=None;assert rc==0,event
  for item in tasks_before_admission(q):task(item)
  # Actual cold inputs and real all-six profiles must exist before any formal fit.
  bindings={'inputs':C.OUT/'inputs/completion.json','profiles_CPU':C.OUT/'all6_profiles_CPU_audit.json','generation_profile':C.OUT/'generation_profile_CPU_audit.json'}
  expected={'inputs':'PASS_COLD_COMPLETE_T13230_DCHECK400_REAL_EEG_INPUT_JOIN_TRANSFORM','profiles_CPU':'PASS_ALL6_REAL_PROFILE_GRADIENTS_AND_SERIALIZED_RESUME_STATES','generation_profile':'PASS_ALL6_REAL_GENERATION_PROFILE_IMAGES_AND_LATENT_LAW'}
  for key,path in bindings.items():
   item=C.read(path);assert item['status']==expected[key] and item['protocol']==C.ref(protocol)
  resources=C.resources(q,starting=True)
  C.write(C.OUT/'actual_formal_admission.json',dict(status='ADMITTED_ONLY_AFTER_ACTUAL_FULL_INPUT_ALL6_PROFILE_AND_CPU_AUDIT',protocol=C.ref(protocol),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),bank=C.bank_state(),resources=resources,**{k:C.ref(p) for k,p in bindings.items()},all_profile_updates_discarded=True,original_ordinary_unchanged=True))
  C.setup(protocol,formal=True)
  for item in tasks_for_fits(q):task(item)
  for item in tasks_for_images(q,80):task(item)
  screen80=C.read(C.OUT/'screens/stage80/summary.json');assert screen80['protocol']==C.ref(protocol)
  if screen80['expand_any']:
   for item in tasks_for_images(q,400):task(item)
  phase='COMPLETE';last=400 if screen80['expand_any'] else 80
  C.write(C.OUT/'completion.json',dict(status='COMPLETE_ALL6_FIXED_EEG_FITS_ALL10_MATCHED_PRIMARY_IMAGE_SCREEN',protocol=C.ref(protocol),bank=C.bank_state(),last_stage=last,last_screen=C.ref(C.OUT/f'screens/stage{last}/summary.json'),all6_formal_new_updates=12420,all6_formal_query_exposures=793800,events=events,seconds=time.monotonic()-started,new_seeds=0,NEW_contacted=False,independent_confirmation_or_SOTA=False,complete_multisubject_semantic_evaluation_still_required=True))
 except BaseException as error:
  # Only this controller's exact owned task may be stopped. Upstream never is.
  if owner is not None and owned is not None:owned.stop_owned(owner,pinned)
  C.write(C.OUT/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),events=events,upstream_not_restarted_or_terminated=True,no_automatic_retry=True));raise

if __name__=='__main__':
 def stop(*unused):raise InterruptedError('Stop only this owned conditional EEG controller')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);run(p.parse_args().protocol)
