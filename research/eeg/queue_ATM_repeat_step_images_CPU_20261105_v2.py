"""Resume only the missing original image stages; original deadline and failures retained."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,subprocess,sys,time,traceback
from pathlib import Path
import atm_repeat_step_image_contract_20261105_v1 as C
from atm_repeat_image_clean_cache_20261105_v2 import release
c=C.setup(sys.argv[1]);q,K,out=c['q'],c['K'],c['out'];assert K.ref(__file__)==q['sources']['controller'];started=time.monotonic();owner=child=None
spec=importlib.util.spec_from_file_location('owned',K.checked(c['physical']['owned_process']));P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
lp=Path(q['code_root'])/'actual_launch.json'
while not lp.exists():
 assert time.monotonic()-started<15;time.sleep(.05)
launch=K.read(lp);assert launch['PID']==os.getpid() and K.process(os.getpid())['start_ticks']==launch['start_ticks']
prior=K.read(K.checked(q['recovery']['original_launch']));assert launch['absolute_deadline_epoch']==prior['absolute_deadline_epoch']
assert not K.process(prior['PID'],prior['start_ticks'])['live']
events=K.read(K.checked(q['recovery']['completed_events']));assert len(events)==5
K.checked(q['recovery']['original_failure']);K.checked(q['recovery']['failed_GPU_receipt'])
for event in events:
 j=K.read(K.checked(event['receipt']));assert j['status']=='COMPLETED' and j['exit_code']==0 and not K.process(j['pid'])['live'] and not K.process(j['wrapper_pid'])['live']

def stage(key,mode,args=(),gpu=False,suffix=None):
 global owner,child
 assert time.time()+935<launch['absolute_deadline_epoch'];name=q['job_prefix']+'_'+(suffix or mode)
 K.write(out/(name+'_cache_advice_controller.json'),release(c['physical']['physical_source']))
 C.resources(c,starting=True)
 cmd=[sys.executable,'-B',q['sources'][key]['path'],'--policy',sys.argv[1],'--mode',mode,*args];receipt=c['repo']/'audit'/(name+'_job.json')
 if gpu:
  assert not receipt.exists();cmd=[sys.executable,'-B',c['physical']['GPU_wrapper']['path'],'--name',name,'--max-seconds','900','--',*cmd]
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0' if gpu else ''
 K.write(out/'progress.json',dict(phase=name,effects_read=False,recovery_version=2))
 with (out/(name+'.log')).open('xb') as f:
  owner=subprocess.Popen(cmd,cwd=c['repo'],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);child=P.capture_wrapper_child(owner,receipt) if gpu else None
  identity=K.process(owner.pid);rc=owner.wait(timeout=935)
 assert rc==0,(name,rc)
 assert not K.process(identity['PID'],identity['start_ticks'])['live']
 if gpu:
  j=K.read(receipt);assert j['status']=='COMPLETED' and j['exit_code']==0 and not K.process(j['pid'])['live'] and not K.process(j['wrapper_pid'])['live']
 events.append(dict(stage=name,GPU=gpu,receipt=K.ref(receipt) if gpu else None));owner=child=None;K.write(out/'recovery_events_v2.json',events)
try:
 stage('GPU','generate',['--domain','R4','--start','320'],True,'generate_R4_320')
 for lo in (0,50,100,150):stage('GPU','generate',['--domain','R80','--start',str(lo)],True,'generate_R80_'+str(lo))
 stage('CPU','audit')
 for d in ['R4','R80']:stage('GPU','encode',['--domain',d],True,'encode_'+d)
 for arm in C.ARMS[3:]:stage('GPU','paper',['--arm',arm],True,'paper_'+arm)
 stage('CPU','score')
 K.checked(q['recovery']['original_failure']);K.checked(q['recovery']['completed_events'])
 K.write(out/'completion.json',dict(status='COMPLETE_PAIRED_REPEAT_STEP_TRAINING_IMAGE_EVALUATION',policy=K.ref(sys.argv[1]),events=events,training=q['training_completion'],
  readout=K.ref(out/'readout.json'),image_audit=K.ref(out/'image_audit.json'),new_images=1800,reused_images=1800,engineering_images=1,new_training_in_image_stage=0,
  seconds=time.monotonic()-started,elapsed_from_original_launch_seconds=time.time()-(prior['absolute_deadline_epoch']-q['maximum_queue_seconds']),
  original_failure=q['recovery']['original_failure'],recovery_version=2,original_deadline_unchanged=True,
  all_registered_metrics_and_contrasts_reported=True,NEW_contacted=False,original_ordinary_unchanged=True,no_automatic_more_fits=True))
 K.write(out/'progress.json',dict(phase='COMPLETE_ALL_FIXED_IMAGES_AND_METRICS'))
except BaseException as exc:
 cleanup=P.stop_owned(owner,child);K.write(out/'recovery_failure_v2.json',dict(error=repr(exc),traceback=traceback.format_exc(),events=events,cleanup=cleanup,seconds=time.monotonic()-started));raise
