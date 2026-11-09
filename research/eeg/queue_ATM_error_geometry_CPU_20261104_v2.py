"""One fixed bounded factorial lifecycle, never retrains or retries failed jobs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import datetime,subprocess,sys,time,traceback
from pathlib import Path
import atm_error_geometry_contract_20261104_v1 as K
path=Path(sys.argv[1]);q=K.read(path);out=Path(q['output_root']);code=path.parent;parent,ctx,e,C=K.setup(q)
repo=Path(ctx['c']['base']['ctx']['relative']['repository_root']);started=time.monotonic();owner=child=None;events=[]
import brainhive_owned_process_CPU_20261029_v1 as P
until=time.monotonic()+10
while not (code/'actual_launch.json').exists():assert time.monotonic()<until;time.sleep(.05)
launch=K.read(code/'actual_launch.json');assert launch['PID']==os.getpid() and K.process(os.getpid())['start_ticks']==launch['start_ticks'];deadline=launch['absolute_deadline_epoch']
def stage(source,label,arguments=(),gpu=False):
    global owner,child
    assert time.time()+930<deadline;K.resources(q,parent,ctx,e,C,starting=True)
    command=[sys.executable,'-B',str(code/q['programs'][source]),'--policy',str(path),*arguments];receipt=repo/'audit'/(label+'_job.json')
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
        job=K.read(receipt);assert job['wrapper_pid']==owner.pid and job['status']=='COMPLETED' and job['exit_code']==0
        assert not K.process(job['pid'])['live'] and not K.process(job['wrapper_pid'])['live']
    events.append(dict(label=label,GPU=gpu,receipt=K.ref(receipt) if gpu else None,exit_code=rc));owner=child=None
def finish(n):
    stage('audit_score',f'audit_images{n}',['--stage',str(n),'--mode','audit'])
    stage('encode',q['job_prefix']+f'_features{n}',['--stage',str(n)],True)
    stage('audit_score',f'score_images{n}',['--stage',str(n),'--mode','score'])
try:
    K.checked(q['reused_statistics']);K.checked(q['reused_profile'])
    assert K.read(q['reused_statistics']['path'])['status']=='COMPLETE_ALL13230_GENUINE_OOF_EEG_ERROR_GEOMETRY_T_ONLY'
    assert K.read(q['reused_profile']['path'])['status']=='PASS_REAL_FOUR_MODELS_ALL_T_STATISTICS_NONZERO_GRADIENTS_AND_CPU_GPU_CLEAN_REPLAY'
    for model in K.MODELS:stage('worker',q['job_prefix']+'_fit_'+model,['--mode','fit','--arm',model],True)
    stage('worker',q['job_prefix']+'_prepare',['--mode','prepare'],True)
    stage('worker',q['job_prefix']+'_CPU',['--mode','CPU'])
    assert K.read(out/'prepared_CPU_audit.json')['status']=='PASS_ALL_FOUR_FINAL_MODELS_OPTIMIZER_RNG_COVERAGE_ALL50_CPU_NEURAL_AND_NUMPY_CLEAN_REPLAY'
    for arm in K.NEW_ARMS:
        stage('generate',q['job_prefix']+f'_gen80_{arm}_0_0',['--stage','80','--arm',arm,'--start','0','--part','0'],True)
    finish(80);readout=K.read(out/'readout80.json');action=readout['interpretation']['action']
    assert action in ('STOP_ALL_NEW_ARMS_JOINT_CLEAR_HARM','EXPAND_ALL_NEW_ARMS_TO400')
    if action=='EXPAND_ALL_NEW_ARMS_TO400':
        for arm in K.NEW_ARMS:
            for begin in (80,160,240,320):
                label=q['job_prefix']+f'_gen400_{arm}_{begin}_0'
                stage('generate',label,['--stage','400','--arm',arm,'--start',str(begin),'--part','0'],True)
                assert K.read(out/(label+'_completion.json'))['status']=='COMPLETE_FIXED_BLOCK'
        finish(400);final=out/'readout400.json'
    else:final=out/'readout80.json'
    K.write(out/'completion.json',dict(status='COMPLETE_NATIVE_ATM_EEG_ERROR_GEOMETRY_PRIORS_PAIRED_IMAGES',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        policy=K.ref(path),final_readout=K.ref(final),initial80_action=action,stage400_completed=final.name=='readout400.json',prepared=K.ref(out/'prepared.json'),events=events,
        seconds=time.monotonic()-started,absolute_deadline_epoch=deadline,new_training_updates=7800,new_models=4,second_innovation_or_SOTA=False,engineering_recovery=q['engineering_recovery'],original_failure_preserved=q['original_failure']))
    K.write(out/'progress.json',dict(phase='COMPLETE_FIXED_ERROR_GEOMETRY_PRIOR_STUDY'))
except BaseException as exc:
    cleanup=P.stop_owned(owner,child);K.write(out/'engineering_recovery_failure_v2.json',dict(status='FAILED_PREFIX_RETAINED_NO_AUTOMATIC_RETRY',error=repr(exc),traceback=traceback.format_exc(),events=events,
        cleanup=cleanup,seconds=time.monotonic()-started,absolute_deadline_epoch=deadline));raise
