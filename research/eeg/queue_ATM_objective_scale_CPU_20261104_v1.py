"""Three missing factorial cells; whole T and full400 before any image effects."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import datetime,subprocess,sys,time,traceback
from pathlib import Path
import atm_objective_scale_contract_20261104_v1 as C
c=C.setup(sys.argv[1]);q=c['q'];K=c['K'];out=c['out'];repo=c['repo'];started=time.monotonic();owner=child=None;events=[]
import brainhive_owned_process_CPU_20261029_v1 as P
until=time.monotonic()+10
while not (Path(q['code_root'])/'actual_launch.json').exists():assert time.monotonic()<until;time.sleep(.05)
launch=K.read(Path(q['code_root'])/'actual_launch.json');assert launch['PID']==os.getpid() and K.process(os.getpid())['start_ticks']==launch['start_ticks'];deadline=launch['absolute_deadline_epoch']
def stage(key,mode,arguments=(),gpu=False,suffix=None):
    global owner,child
    assert time.time()+930<deadline;C.resources(c,starting=True);label=q['job_prefix']+'_'+(suffix or mode)
    command=[sys.executable,'-B',q['sources'][key]['path'],'--policy',sys.argv[1],'--mode',mode,*arguments];receipt=repo/'audit'/(label+'_job.json')
    if gpu:
        assert not receipt.exists() and not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        command=[sys.executable,'-B',str(repo/'run_gpu_job_v2.py'),'--name',label,'--max-seconds','900','--',*command]
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0' if gpu else '';env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    K.write(out/'progress.json',dict(phase=label,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    with (out/(label+'.log')).open('xb') as log:
        owner=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);child=P.capture_wrapper_child(owner,receipt) if gpu else None;rc=owner.wait(timeout=930)
    assert rc==0,dict(label=label,exit_code=rc)
    if gpu:
        j=K.read(receipt);assert j['status']=='COMPLETED' and j['exit_code']==0 and j['wrapper_pid']==owner.pid and not K.process(j['pid'])['live'] and not K.process(j['wrapper_pid'])['live']
    events.append(dict(label=label,GPU=gpu,exit_code=rc,receipt=K.ref(receipt) if gpu else None));owner=child=None
def finish(n):
    stage('CPU','audit',['--stage',str(n)],suffix='audit_'+str(n));stage('GPU','encode',['--stage',str(n)],True,'encode_'+str(n));stage('CPU','score',['--stage',str(n)],suffix='score_'+str(n))
try:
    stage('GPU','profile',gpu=True)
    for arm in C.MODELS:stage('GPU','fit',['--arm',arm],True,'fit_'+arm)
    stage('GPU','prepare',gpu=True);stage('CPU','conditions')
    for lo in (0,80,160,240,320):stage('GPU','generate',['--start',str(lo)],True,'generate_'+str(lo))
    finish(400)
    final=400
    K.write(out/'completion.json',dict(status='COMPLETE_NATIVE_EEG_OBJECTIVE_SCALE_FACTORIAL_FULL400',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),policy=K.ref(sys.argv[1]),
        final_readout=K.ref(out/f'readout{final}.json'),prepared=K.ref(out/'prepared.json'),all400_before_effect_read=True,stage400_completed=final==400,events=events,seconds=time.monotonic()-started,absolute_deadline_epoch=deadline,
        new_models=3,new_updates=2490,new_exposures=396900,ordinary_unchanged=True,second_independent_method_or_SOTA=False))
    K.write(out/'progress.json',dict(phase='COMPLETE_FIXED_EEG_TRAINING_TARGET_DIAGNOSTIC'))
except BaseException as exc:
    cleanup=P.stop_owned(owner,child);K.write(out/'failure.json',dict(status='FAILED_PREFIX_RETAINED_NO_AUTO_RETRY',error=repr(exc),traceback=traceback.format_exc(),events=events,cleanup=cleanup,seconds=time.monotonic()-started,absolute_deadline_epoch=deadline));raise
