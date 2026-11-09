"""Finish only unrun image/evaluation slots; reuse all fits and complete CPU replay."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import datetime,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[1]).parent))
import enigma_native_reference_contract_20261104_v1 as C
c=C.setup(sys.argv[1]);q=c['q'];K=c['K'];out=c['out'];repo=c['repo'];started=time.monotonic();owner=child=None
am=K.read(sys.argv[2]);assert am['original_policy']==K.ref(sys.argv[1])
assert am['sources']['controller']==K.ref(__file__)
events=list(am['completed_training_prepare_CPU_events'])
for event in events:
    if event['GPU']:
        j=K.read(K.checked(event['receipt']));assert j['exit_code']==0 and j['status']=='COMPLETED'
        assert not K.process(j['pid'])['live'] and not K.process(j['wrapper_pid'])['live']
assert K.read(K.checked(am['CPU_replay']))['status']=='PASS_NATIVE63_COMPLETE_OOF_AND_D_CPU_REPLAY'
import brainhive_owned_process_CPU_20261029_v1 as P
until=time.monotonic()+10;launch_path=Path(sys.argv[2]).parent/'actual_recovery_launch.json'
while not launch_path.exists():assert time.monotonic()<until;time.sleep(.05)
launch=K.read(launch_path);assert launch['PID']==os.getpid() and K.process(os.getpid())['start_ticks']==launch['start_ticks']
deadline=launch['absolute_deadline_epoch']
def stage(key,mode,arguments=(),gpu=False,suffix=None):
    global owner,child
    assert time.time()+930<deadline;C.resources(c,starting=True)
    label=am['job_prefix']+'_'+(suffix or mode)
    command=[sys.executable,'-B',am['sources'][key]['path'],'--policy',sys.argv[1],'--mode',mode,*arguments]
    receipt=repo/'audit'/(label+'_job.json')
    if gpu:
        command+=['--amendment',sys.argv[2]]
        assert not receipt.exists() and not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        command=[sys.executable,'-B',str(repo/'run_gpu_job_v2.py'),'--name',label,'--max-seconds','900','--',*command]
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0' if gpu else '';env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    K.write(out/'progress.json',dict(phase=label,engineering_recovery=K.ref(sys.argv[2]),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    with (out/(label+'.log')).open('xb') as log:
        owner=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        child=P.capture_wrapper_child(owner,receipt) if gpu else None;rc=owner.wait(timeout=930)
    assert rc==0,dict(label=label,exit_code=rc)
    if gpu:
        j=K.read(receipt);assert j['status']=='COMPLETED' and j['exit_code']==0 and j['wrapper_pid']==owner.pid
        assert not K.process(j['pid'])['live'] and not K.process(j['wrapper_pid'])['live']
    events.append(dict(label=label,GPU=gpu,exit_code=rc,receipt=K.ref(receipt) if gpu else None));owner=child=None
try:
    for lo in (0,80,160,240,320):stage('GPU','generate',['--start',str(lo)],True,'generate_'+str(lo))
    stage('CPU','audit');stage('GPU','encode',gpu=True);stage('CPU','score')
    assert K.ref(am['retained_original_failure']['path'])==am['retained_original_failure']
    K.write(out/'completion.json',dict(status='COMPLETE_NATIVE63_FULL150_OOF_AND_FULL400_REFERENCE',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),policy=K.ref(sys.argv[1]),events=events,
        prepared=K.ref(out/'prepared.json'),final_readout=K.ref(out/'readout400.json'),
        new_models=3,new_updates=7800,new_exposures=3969000,new_images=400,reused_images=2400,
        seconds=time.monotonic()-started,absolute_deadline_epoch=deadline,all400_before_effect_reads=True,
        engineering_recovery=K.ref(sys.argv[2]),actual_recovery_launch=K.ref(launch_path),
        original_failure_retained=True,no_training_or_condition_export_repeated=True,
        original_ordinary_unchanged=True,published_reference_NOT_novel_method=True,second_independent_method_or_SOTA=False))
    K.write(out/'progress.json',dict(phase='COMPLETE_NATIVE63_REFERENCE',engineering_recovery=K.ref(sys.argv[2])))
except BaseException as exc:
    cleanup=P.stop_owned(owner,child)
    K.write(out/'runtime_v7_recovery_failure.json',dict(status='FAILED_PREFIX_RETAINED_NO_AUTO_RETRY',error=repr(exc),
        traceback=traceback.format_exc(),events=events,cleanup=cleanup,seconds=time.monotonic()-started,
        absolute_deadline_epoch=deadline,amendment=K.ref(sys.argv[2])));raise
