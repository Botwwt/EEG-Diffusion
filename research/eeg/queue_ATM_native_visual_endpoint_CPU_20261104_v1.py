"""One finite OLD-only diagnostic. No fit, auto expansion, or automatic retry."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import datetime,subprocess,sys,time,traceback
from pathlib import Path
import atm_native_visual_endpoint_contract_20261104_v1 as V
c=V.setup(sys.argv[1]);q=c['q'];K=c['K'];out=c['out'];repo=c['repo'];code=Path(q['code_root']);started=time.monotonic();owner=child=None;events=[]
import brainhive_owned_process_CPU_20261029_v1 as P
until=time.monotonic()+10
while not (code/'actual_launch.json').exists():assert time.monotonic()<until;time.sleep(.05)
launch=K.read(code/'actual_launch.json');assert launch['PID']==os.getpid() and K.process(os.getpid())['start_ticks']==launch['start_ticks'];deadline=launch['absolute_deadline_epoch']
def stage(key,mode,gpu=False):
    global owner,child
    assert time.time()+930<deadline;V.resources(c,starting=True);label=q['job_prefix']+'_'+mode
    command=[sys.executable,'-B',q['sources'][key]['path'],'--policy',sys.argv[1],'--mode',mode];receipt=repo/'audit'/(label+'_job.json')
    if gpu:
        assert not receipt.exists() and not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        command=[sys.executable,'-B',str(repo/'run_gpu_job_v2.py'),'--name',label,'--max-seconds','900','--',*command]
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0' if gpu else '';env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    K.write(out/'progress.json',dict(phase=mode,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    with (out/(mode+'.log')).open('xb') as log:
        owner=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        child=P.capture_wrapper_child(owner,receipt) if gpu else None;rc=owner.wait(timeout=930)
    assert rc==0,dict(mode=mode,exit_code=rc)
    if gpu:
        job=K.read(receipt);assert job['status']=='COMPLETED' and job['exit_code']==0 and job['wrapper_pid']==owner.pid
        assert not K.process(job['pid'])['live'] and not K.process(job['wrapper_pid'])['live']
    events.append(dict(mode=mode,GPU=gpu,exit_code=rc,receipt=K.ref(receipt) if gpu else None));owner=child=None
try:
    stage('CPU','prepare');stage('GPU','generate',True);stage('CPU','audit');stage('GPU','encode',True);stage('CPU','score')
    K.write(out/'completion.json',dict(status='COMPLETE_PRIVILEGED_NATIVE_ATM_ENDPOINT_NO_METHOD_GAIN_CLAIM',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        policy=K.ref(sys.argv[1]),readout=K.ref(out/'readout80.json'),prepared=K.ref(out/'prepared.json'),image_audit=K.ref(out/'image_audit.json'),events=events,
        seconds=time.monotonic()-started,absolute_deadline_epoch=deadline,new_models=0,new_updates=0,new_research_images=160,reused_baseline_images=160,automatic_fit_or_expansion=False,second_method_or_SOTA=False))
    K.write(out/'progress.json',dict(phase='COMPLETE_FIXED_VISUAL_ENDPOINT'))
except BaseException as exc:
    cleanup=P.stop_owned(owner,child)
    K.write(out/'failure.json',dict(status='FAILED_PREFIX_RETAINED_NO_AUTOMATIC_RETRY',error=repr(exc),traceback=traceback.format_exc(),events=events,cleanup=cleanup,
        seconds=time.monotonic()-started,absolute_deadline_epoch=deadline));raise
