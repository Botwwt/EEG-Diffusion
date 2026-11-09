"""Finite baseline completion; no fits, no automatic retry, OLD only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import datetime,subprocess,sys,time,traceback
from pathlib import Path
import atm_parent_R80_contract_20261105_v174 as C
c=C.setup(sys.argv[1]);q,K,out=c['q'],c['K'],c['out'];assert K.ref(__file__)==q['sources']['controller'];started=time.monotonic();owner=child=None;events=[]
import brainhive_owned_process_CPU_20261029_v1 as P
from atm_repeat_image_clean_cache_20261105_v2 import release
lp=Path(q['code_root'])/'actual_launch.json'
while not lp.exists():assert time.monotonic()-started<15;time.sleep(.05)
launch=K.read(lp);assert launch['PID']==os.getpid() and K.process(os.getpid())['start_ticks']==launch['start_ticks']
def stage(mode,gpu=False,lo=None):
 global owner,child
 assert time.time()+935<launch['absolute_deadline_epoch'];release(c['physical']['physical_source']);C.resources(c,True)
 suffix=mode+('_'+str(lo) if lo is not None else '');name=q['job_prefix']+'_'+suffix;receipt=c['repo']/'audit'/(name+'_job.json')
 cmd=[sys.executable,'-B',q['sources']['runner']['path'],'--policy',sys.argv[1],'--mode',mode]+([] if lo is None else ['--start',str(lo)])
 if gpu:
  assert not receipt.exists();cmd=[sys.executable,'-B',c['physical']['GPU_wrapper']['path'],'--name',name,'--max-seconds','900','--',*cmd]
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0' if gpu else ''
 K.write(out/'progress.json',dict(phase=name,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),effects_read=False))
 with (out/(name+'.log')).open('xb') as f:
  owner=subprocess.Popen(cmd,cwd=c['repo'],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);child=P.capture_wrapper_child(owner,receipt) if gpu else None
  ident=K.process(owner.pid);rc=owner.wait(timeout=935)
 assert rc==0,(name,rc);assert not K.process(ident['PID'],ident['start_ticks'])['live']
 if gpu:
  j=K.read(receipt);assert j['status']=='COMPLETED' and j['exit_code']==0
  assert not K.process(j['pid'])['live'] and not K.process(j['wrapper_pid'])['live']
 events.append(dict(stage=name,GPU=gpu,receipt=K.ref(receipt) if gpu else None));owner=child=None;K.write(out/'events.json',events)
try:
 stage('conditions',True);stage('CPU_conditions');stage('profile',True)
 for lo in (0,100):stage('generate',True,lo)
 stage('audit');stage('encode',True);stage('paper',True);stage('score')
 K.write(out/'completion.json',dict(status='COMPLETE_EXACT_ORIGINAL_ATMS_R80_PAIRED_BASELINE',policy=K.ref(sys.argv[1]),events=events,
  readout=K.ref(out/'score_completion.json'),image_audit=K.ref(out/'audit_completion.json'),new_images=200,reused_images=200,new_updates=0,
  seconds=time.monotonic()-started,NEW_contacted=False,ordinary_unchanged=True,no_automatic_more_fits=True))
 K.write(out/'progress.json',dict(phase='COMPLETE_ALL400_R80_AND15_METRICS'))
except BaseException as e:
 cleanup=P.stop_owned(owner,child);K.write(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),events=events,cleanup=cleanup));raise
