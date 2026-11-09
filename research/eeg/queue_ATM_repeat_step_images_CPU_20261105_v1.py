"""Same reserved project, finite complete paired image lifecycle."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,subprocess,sys,time,traceback
from pathlib import Path
import atm_repeat_step_image_contract_20261105_v1 as C
c=C.setup(sys.argv[1]);q,K,out=c['q'],c['K'],c['out'];assert K.ref(__file__)==q['sources']['controller'];started=time.monotonic();owner=child=None;events=[]
spec=importlib.util.spec_from_file_location('owned',K.checked(c['physical']['owned_process']));P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
lp=Path(q['code_root'])/'actual_launch.json'
while not lp.exists():
 assert time.monotonic()-started<15;time.sleep(.05)
launch=K.read(lp);assert launch['PID']==os.getpid() and K.process(os.getpid())['start_ticks']==launch['start_ticks']
def stage(key,mode,args=(),gpu=False,suffix=None):
 global owner,child
 assert time.time()+935<launch['absolute_deadline_epoch'];C.resources(c,starting=True);name=q['job_prefix']+'_'+(suffix or mode)
 cmd=[sys.executable,'-B',q['sources'][key]['path'],'--policy',sys.argv[1],'--mode',mode,*args];receipt=c['repo']/'audit'/(name+'_job.json')
 if gpu:
  assert not receipt.exists();cmd=[sys.executable,'-B',c['physical']['GPU_wrapper']['path'],'--name',name,'--max-seconds','900','--',*cmd]
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0' if gpu else ''
 K.write(out/'progress.json',dict(phase=name,effects_read=False))
 with (out/(name+'.log')).open('xb') as f:
  owner=subprocess.Popen(cmd,cwd=c['repo'],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);child=P.capture_wrapper_child(owner,receipt) if gpu else None
  identity=K.process(owner.pid);rc=owner.wait(timeout=935)
 assert rc==0,(name,rc)
 assert not K.process(identity['PID'],identity['start_ticks'])['live']
 if gpu:
  j=K.read(receipt);assert j['status']=='COMPLETED' and j['exit_code']==0 and not K.process(j['pid'])['live'] and not K.process(j['wrapper_pid'])['live']
 events.append(dict(stage=name,GPU=gpu,receipt=K.ref(receipt) if gpu else None));owner=child=None;K.write(out/'events.json',events)
try:
 stage('GPU','profile',gpu=True)
 for domain,stops in [('R4',(0,80,160,240,320)),('R80',(0,50,100,150))]:
  for lo in stops:stage('GPU','generate',['--domain',domain,'--start',str(lo)],True,'generate_'+domain+'_'+str(lo))
 stage('CPU','audit')
 for d in ['R4','R80']:stage('GPU','encode',['--domain',d],True,'encode_'+d)
 for arm in C.ARMS[3:]:stage('GPU','paper',['--arm',arm],True,'paper_'+arm)
 stage('CPU','score')
 K.write(out/'completion.json',dict(status='COMPLETE_PAIRED_REPEAT_STEP_TRAINING_IMAGE_EVALUATION',policy=K.ref(sys.argv[1]),events=events,training=q['training_completion'],
  readout=K.ref(out/'readout.json'),image_audit=K.ref(out/'image_audit.json'),new_images=1800,reused_images=1800,engineering_images=1,new_training_in_image_stage=0,
  seconds=time.monotonic()-started,all_registered_metrics_and_contrasts_reported=True,NEW_contacted=False,original_ordinary_unchanged=True,no_automatic_more_fits=True))
 K.write(out/'progress.json',dict(phase='COMPLETE_ALL_FIXED_IMAGES_AND_METRICS'))
except BaseException as exc:
 cleanup=P.stop_owned(owner,child);K.write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),events=events,cleanup=cleanup,seconds=time.monotonic()-started));raise
