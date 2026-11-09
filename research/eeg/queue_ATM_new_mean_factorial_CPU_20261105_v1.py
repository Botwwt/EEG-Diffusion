"""Finite OLD-only full400 diagnostic, no retry or intermediate score selection."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import datetime,subprocess,sys,time,traceback
from pathlib import Path
import atm_new_mean_factorial_contract_20261105_v1 as C
c=C.setup(sys.argv[1]);q,K,out=c['q'],c['K'],c['out'];started=time.monotonic();owner=child=None;events=[]
import brainhive_owned_process_CPU_20261029_v1 as P
until=time.monotonic()+10
while not (Path(q['code_root'])/'actual_launch.json').exists():
 assert time.monotonic()<until;time.sleep(.05)
launch=K.read(Path(q['code_root'])/'actual_launch.json');assert launch['PID']==os.getpid() and K.process(os.getpid())['start_ticks']==launch['start_ticks']
deadline=launch['absolute_deadline_epoch']
def stage(key,mode,args=(),gpu=False,suffix=None):
 global owner,child
 assert time.time()+930<deadline;C.resources(c)
 label=q['job_prefix']+'_'+(suffix or mode);receipt=c['repo']/'audit'/(label+'_job.json')
 cmd=[sys.executable,'-B',q['sources'][key]['path'],'--policy',sys.argv[1],'--mode',mode,*args]
 if gpu:
  assert not receipt.exists();cmd=[sys.executable,'-B',str(c['repo']/'run_gpu_job_v2.py'),'--name',label,'--max-seconds','900','--',*cmd]
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0' if gpu else '';env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
 K.write(out/'progress.json',dict(phase=label,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 with (out/(label+'.log')).open('xb') as log:
  owner=subprocess.Popen(cmd,cwd=c['repo'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  child=P.capture_wrapper_child(owner,receipt) if gpu else None;rc=owner.wait(timeout=930)
 assert rc==0,dict(label=label,exit_code=rc)
 if gpu:
  job=K.read(receipt);assert job['status']=='COMPLETED' and job['exit_code']==0 and job['wrapper_pid']==owner.pid
  assert not K.process(job['pid'])['live'] and not K.process(job['wrapper_pid'])['live']
 events.append(dict(stage=label,GPU=gpu,exit_code=rc,receipt=K.ref(receipt) if gpu else None));owner=child=None
try:
 stage('CPU','prepare')
 for lo in (0,80,160,240,320):stage('GPU','generate',['--start',str(lo)],True,'generate_'+str(lo))
 stage('CPU','audit');stage('GPU','encode',gpu=True);stage('CPU','score')
 K.write(out/'completion.json',dict(status='COMPLETE_NEW_MEAN_FACTORIAL_FULL400',policy=K.ref(sys.argv[1]),readout=K.ref(out/'readout.json'),
  image_audit=K.ref(out/'image_audit.json'),events=events,seconds=time.monotonic()-started,new_images=1200,reused_images=400,new_updates=0,all400_before_effect_readout=True))
 K.write(out/'progress.json',dict(phase='COMPLETE_FULL400'))
except BaseException as e:
 cleanup=P.stop_owned(owner,child);K.write(out/'failure.json',dict(status='FAILED_PREFIX_RETAINED_NO_AUTO_RETRY',error=repr(e),traceback=traceback.format_exc(),events=events,cleanup=cleanup));raise
