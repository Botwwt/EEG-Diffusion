"""One finite OLD-only image study; all arms complete before metric readout."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import datetime,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[1]).parent))
import enigma_R80_count_image_contract_20261104_v1 as C
c=C.setup(sys.argv[1]);q=c['q'];K=c['K'];out=c['out'];repo=c['repo'];started=time.monotonic();owner=child=None;events=[]
import brainhive_owned_process_CPU_20261029_v1 as P
until=time.monotonic()+10;launch_path=Path(sys.argv[1]).parent/'actual_launch.json'
while not launch_path.exists():assert time.monotonic()<until;time.sleep(.05)
launch=K.read(launch_path);assert launch['PID']==os.getpid() and K.process(os.getpid())['start_ticks']==launch['start_ticks']
deadline=launch['absolute_deadline_epoch']
def stage(key,mode,arguments=(),gpu=False,suffix=None):
    global owner,child
    assert time.time()+930<deadline;C.resources(c,starting=True)
    label=q['job_prefix']+'_'+(suffix or mode)
    command=[sys.executable,'-B',q['sources'][key]['path'],'--policy',sys.argv[1],'--mode',mode,*arguments]
    receipt=repo/'audit'/(label+'_job.json')
    if gpu:
        assert not receipt.exists() and not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        command=[sys.executable,'-B',str(repo/'run_gpu_job_v2.py'),'--name',label,'--max-seconds','900','--',*command]
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0' if gpu else '';env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    K.write(out/'progress.json',dict(phase=label,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),effects_read=False))
    with (out/(label+'.log')).open('xb') as log:
        owner=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        child=P.capture_wrapper_child(owner,receipt) if gpu else None;rc=owner.wait(timeout=930)
    assert rc==0,dict(label=label,exit_code=rc)
    if gpu:
        j=K.read(receipt);assert j['status']=='COMPLETED' and j['exit_code']==0 and j['wrapper_pid']==owner.pid
        assert not K.process(j['pid'])['live'] and not K.process(j['wrapper_pid'])['live']
    events.append(dict(label=label,GPU=gpu,exit_code=rc,receipt=K.ref(receipt) if gpu else None));owner=child=None
try:
    stage('GPU','profile',gpu=True)
    for lo in (0,50,100,150):stage('GPU','generate',['--start',str(lo)],True,'generate_'+str(lo))
    stage('CPU','audit');stage('GPU','encode',gpu=True)
    for arm in C.ARMS:stage('GPU','paper',['--arm',arm],True,'paper_'+arm)
    stage('CPU','score')
    K.write(out/'completion.json',dict(status='COMPLETE_FOUR_ARM_R80_COUNT_IMAGE_STUDY',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),policy=K.ref(sys.argv[1]),events=events,
        readout=K.ref(out/'readout.json'),conditions=q['measurement_conditions'],image_audit=K.ref(out/'image_audit.json'),
        new_models=0,new_updates=0,new_images=800,engineering_images=1,
        seconds=time.monotonic()-started,absolute_deadline_epoch=deadline,all800_before_effect_read=True,
        original_ordinary_unchanged=True,NEW_contacted=False,no_automatic_followup_fit=True,originality_or_SOTA_not_established=True))
    K.write(out/'progress.json',dict(phase='COMPLETE_FOUR_ARM_R80_COUNT_IMAGE_STUDY'))
except BaseException as exc:
    cleanup=P.stop_owned(owner,child)
    K.write(out/'failure.json',dict(status='FAILED_PREFIX_RETAINED_NO_AUTO_RETRY',error=repr(exc),traceback=traceback.format_exc(),
        events=events,cleanup=cleanup,seconds=time.monotonic()-started,absolute_deadline_epoch=deadline));raise
