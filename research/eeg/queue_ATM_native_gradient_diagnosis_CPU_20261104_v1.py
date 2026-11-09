"""One bounded dependency wait, then one diagnostic GPU job; no fits/retries."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import datetime,fcntl,hashlib,json,subprocess,sys,time,traceback
from pathlib import Path
q=json.loads(Path(sys.argv[1]).read_bytes());root=Path(q['project_root']);out=Path(q['output_root']);code=Path(q['code_root']);owner=child=None
def read(p):return json.loads(Path(p).read_bytes())
def write(path,value):
    p=Path(path);tmp=p.with_name(p.name+'.pending');assert not tmp.exists();tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)
def process(pid,ticks=None):
    try:s=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return dict(PID=pid,state='MISSING',live=False)
    return dict(PID=pid,start_ticks=int(s[19]),state=s[0],live=s[0] not in ('Z','X') and (ticks is None or int(s[19])==ticks))
until=time.monotonic()+10
while not (code/'actual_launch.json').exists():assert time.monotonic()<until;time.sleep(.05)
launch=read(code/'actual_launch.json');assert launch['PID']==os.getpid() and process(os.getpid())['start_ticks']==launch['start_ticks'];deadline=launch['absolute_deadline_epoch'];started=time.monotonic()
repo=Path(q['repository_root']);dep=read(q['dependency_launch']['path']);depout=Path(q['dependency_output_root'])
try:
    while True:
        assert time.time()+950<deadline
        current=process(dep['PID'],dep['start_ticks'])
        if current['live']:
            write(out/'progress.json',dict(phase='WAIT_EXACT_WHOLE_IMAGE_COMPARISON_OWNER',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),dependency_owner=current,no_Torch_or_CUDA_imported=True));time.sleep(15);continue
        assert not (depout/'failure.json').exists(),read(depout/'failure.json') if (depout/'failure.json').exists() else None
        done=read(depout/'completion.json');assert done['status']=='COMPLETE_NATIVE_EEG_CONDITIONAL_STIMULUS_TARGET_DIAGNOSTIC' and done['policy']==q['study_policy']
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
            write(out/'progress.json',dict(phase='WAIT_GPU_EMPTY_NO_PREEMPTION',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()));time.sleep(15);continue
        break
    sys.path.insert(0,str(Path(q['study_policy']['path']).parent));import atm_conditional_stimulus_contract_20261104_v1 as C
    c=C.setup(q['study_policy']['path']);K=c['K']
    for pin in q['sources'].values():K.checked(pin)
    for item in q['states'].values():K.checked(item['checkpoint'])
    for event in done['events']:
        assert event['exit_code']==0
        if event['GPU']:
            j=read(K.checked(event['receipt']));assert j['status']=='COMPLETED' and j['exit_code']==0 and not process(j['pid'])['live'] and not process(j['wrapper_pid'])['live']
    resources=C.resources(c,starting=True);used=c['cv']['ctx']['base']['ctx']['C'].allocated(root);required=resources['required_including_this_project']+q['maximum_output_bytes']-used;assert used<=q['maximum_output_bytes'] and resources['free_bytes']>=required
    locks=[]
    for name in ('GPU_JOB.lock','TRIALSET_GPU_INTERLEAVE.lock'):
        f=(repo/'audit'/name).open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==c['parent']['GPU_UUID']
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    write(code/'execution_admission.json',dict(status='ADMIT_FROZEN_ALL_T_GRADIENT_DIAGNOSIS_AFTER_WHOLE_EXIT',whole_dependency_completion=K.ref(depout/'completion.json'),actual_dependency_owner=current,
        original_image_study_sources_and_all_four_checkpoints_checked=True,resources=resources,required_including_new192MiB=required,no_new_fits_or_images=True,all_previous_forecasts_retained=True))
    for f in locks:fcntl.flock(f,fcntl.LOCK_UN);f.close()
    import brainhive_owned_process_CPU_20261029_v1 as P
    receipt=repo/'audit'/(q['job_name']+'_job.json');assert not receipt.exists();env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0';env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    command=[sys.executable,'-B',str(repo/'run_gpu_job_v2.py'),'--name',q['job_name'],'--max-seconds','900','--',sys.executable,'-B',q['sources']['GPU']['path'],'--policy',sys.argv[1]]
    write(out/'progress.json',dict(phase='RUNNING_FROZEN_ALL_T_DIAGNOSTIC',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    with (out/'GPU.log').open('xb') as log:
        owner=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);child=P.capture_wrapper_child(owner,receipt);rc=owner.wait(timeout=930)
    assert rc==0,rc;j=read(receipt);assert j['status']=='COMPLETED' and j['exit_code']==0 and j['wrapper_pid']==owner.pid and not process(j['pid'])['live'] and not process(j['wrapper_pid'])['live']
    worker=read(out/'completion.json');assert worker['status']=='COMPLETE_FROZEN_FOUR_STATE_ALL_T_COARSE_FINE_GRADIENT_DIAGNOSIS'
    write(out/'controller_completion.json',dict(status='COMPLETE_BOUNDED_ALL_T_GRADIENT_DIAGNOSIS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),worker=K.ref(out/'completion.json'),job=K.ref(receipt),seconds=time.monotonic()-started,absolute_deadline_epoch=deadline,new_fits_or_images=0))
    write(out/'progress.json',dict(phase='COMPLETE_FROZEN_ALL_T_DIAGNOSIS'))
except BaseException as exc:
    cleanup=None
    if owner is not None:
        import brainhive_owned_process_CPU_20261029_v1 as P
        cleanup=P.stop_owned(owner,child)
    write(out/'controller_failure.json',dict(status='FAILED_DEPENDENCY_OR_DIAGNOSTIC_NO_RETRY',error=repr(exc),traceback=traceback.format_exc(),cleanup=cleanup,absolute_deadline_epoch=deadline));raise
