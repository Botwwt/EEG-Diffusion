"""One finite native EEG reference lifecycle; never retry failed science automatically."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import datetime,subprocess,sys,time,traceback
from pathlib import Path
import enigma_native_reference_contract_20261104_v1 as C
c=C.setup(sys.argv[1]);q=c['q'];K=c['K'];out=c['out'];repo=c['repo'];started=time.monotonic();owner=child=None;events=[]
import brainhive_owned_process_CPU_20261029_v1 as P
until=time.monotonic()+10
while not (Path(q['code_root'])/'actual_launch.json').exists():assert time.monotonic()<until;time.sleep(.05)
launch=K.read(Path(q['code_root'])/'actual_launch.json')
assert launch['PID']==os.getpid() and K.process(os.getpid())['start_ticks']==launch['start_ticks']
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
    K.write(out/'progress.json',dict(phase=label,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    with (out/(label+'.log')).open('xb') as log:
        owner=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        child=P.capture_wrapper_child(owner,receipt) if gpu else None;rc=owner.wait(timeout=930)
    assert rc==0,dict(label=label,exit_code=rc)
    if gpu:
        j=K.read(receipt)
        assert j['status']=='COMPLETED' and j['exit_code']==0 and j['wrapper_pid']==owner.pid
        assert not K.process(j['pid'])['live'] and not K.process(j['wrapper_pid'])['live']
    events.append(dict(label=label,GPU=gpu,exit_code=rc,receipt=K.ref(receipt) if gpu else None));owner=child=None
try:
    if 'resume_origin_policy' in q:
        assert K.read(K.checked(q['reused_profile']))['author_train_epoch_one_update_state_max_error']==0.
        events.extend(q['reused_successful_events'])
        for event in events:
            j=K.read(K.checked(event['receipt']))
            assert j['status']=='COMPLETED' and j['exit_code']==0 and not K.process(j['pid'])['live'] and not K.process(j['wrapper_pid'])['live']
    else:stage('GPU','profile',gpu=True)
    for role in C.ROLES:
        for end in (50,100,150):
            if 'resume_origin_policy' in q and role=='full' and end==50:continue
            stage('GPU','fit',['--role',role,'--end-epoch',str(end)],True,'fit_'+role+'_'+str(end))
    stage('GPU','prepare',gpu=True);stage('CPU','conditions')
    for lo in (0,80,160,240,320):stage('GPU','generate',['--start',str(lo)],True,'generate_'+str(lo))
    stage('CPU','audit');stage('GPU','encode',gpu=True);stage('CPU','score')
    K.write(out/'completion.json',dict(status='COMPLETE_NATIVE63_FULL150_OOF_AND_FULL400_REFERENCE',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),policy=K.ref(sys.argv[1]),events=events,
        prepared=K.ref(out/'prepared.json'),final_readout=K.ref(out/'readout400.json'),
        new_models=3,new_updates=7800,new_exposures=3969000,new_images=400,reused_images=2400,
        seconds=time.monotonic()-started,absolute_deadline_epoch=deadline,all400_before_effect_reads=True,
        original_ordinary_unchanged=True,published_reference_NOT_novel_method=True,second_independent_method_or_SOTA=False))
    K.write(out/'progress.json',dict(phase='COMPLETE_NATIVE63_REFERENCE'))
except BaseException as exc:
    cleanup=P.stop_owned(owner,child)
    K.write(out/'failure.json',dict(status='FAILED_PREFIX_RETAINED_NO_AUTO_RETRY',error=repr(exc),
        traceback=traceback.format_exc(),events=events,cleanup=cleanup,seconds=time.monotonic()-started,absolute_deadline_epoch=deadline));raise
