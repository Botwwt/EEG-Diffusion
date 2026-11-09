"""Finite 3-arm full-T training and EEG-only condition exports; images next."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,subprocess,sys,time,traceback
from pathlib import Path
import atm_repeat_step_training_contract_20261105_v1 as C
c=C.setup(sys.argv[1]);q,K,out,pq=c['q'],c['K'],c['out'],c['physical'];assert K.ref(__file__)==q['sources']['controller']
spec=importlib.util.spec_from_file_location('owned',K.checked(pq['owned_process']));P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
started=time.monotonic();owner=child=None;events=[];lp=Path(q['code_root'])/'actual_launch.json'
while not lp.exists():
 assert time.monotonic()-started<15;time.sleep(.05)
la=K.read(lp);assert la['PID']==os.getpid() and K.process(os.getpid())['start_ticks']==la['start_ticks']
def gpu(mode,arm='',end=0):
 global owner,child
 C.resources(c,starting=True);name=q['job_prefix']+'_'+mode+('_'+arm+'_'+str(end) if mode=='fit' else '')
 receipt=Path(pq['GPU_wrapper']['path']).parent/'audit'/(name+'_job.json');assert not receipt.exists()
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
 cmd=[sys.executable,'-B',pq['GPU_wrapper']['path'],'--name',name,'--max-seconds','900','--',sys.executable,'-B',q['sources']['GPU']['path'],'--policy',sys.argv[1],'--mode',mode]
 if arm:cmd+=['--arm',arm,'--end',str(end)]
 with (out/(name+'.log')).open('xb') as log:
  owner=subprocess.Popen(cmd,cwd=receipt.parent.parent,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  child=P.capture_wrapper_child(owner,receipt);rc=owner.wait(timeout=min(935,la['absolute_deadline_epoch']-time.time()))
 j=K.read(receipt);assert rc==0 and j['status']=='COMPLETED' and j['exit_code']==0
 assert not K.process(j['pid'])['live'] and not K.process(j['wrapper_pid'])['live'];owner=child=None
 events.append(dict(mode=mode,arm=arm,end=end,GPU=True,receipt=K.ref(receipt)));K.write(out/'events.json',events)
try:
 for arm in q['arms']:
  for end in (5,10):gpu('fit',arm,end)
 gpu('conditions');C.resources(c,starting=True)
 with (out/'CPU.log').open('xb') as log:
  owner=subprocess.Popen([sys.executable,'-B',q['sources']['CPU']['path'],sys.argv[1]],cwd=q['code_root'],env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  identity=K.process(owner.pid);K.write(out/'CPU_owner.json',identity);assert owner.wait(timeout=min(900,la['absolute_deadline_epoch']-time.time()))==0
 assert not K.process(identity['PID'],identity['start_ticks'])['live'];owner=None
 audit=K.read(out/'independent_CPU_audit.json');assert audit['status']=='PASS_ALL3600_CPU_CONDITIONS_AND_ALL30_EPOCHS'
 K.write(out/'training_completion.json',dict(status='COMPLETE_THREE_ARM_T10_AND_ALL3600_CONDITIONS_IMAGES_PENDING',policy=K.ref(sys.argv[1]),events=events,CPU_audit=K.ref(out/'independent_CPU_audit.json'),
  new_models=3,new_updates=2490,main_query_exposures=396900,view_exposures=1984500,new_images=0,seconds=time.monotonic()-started,
  whole_image_budget_reserved=True,image_executor_NOT_started=True,no_automatic_extra_fit=True))
except BaseException as exc:
 cleanup=P.stop_owned(owner,child);K.write(out/'controller_failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),cleanup=cleanup,events=events,seconds=time.monotonic()-started));raise
