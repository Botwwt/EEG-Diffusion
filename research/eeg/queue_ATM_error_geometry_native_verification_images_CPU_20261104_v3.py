"""Replay frozen conditions in native units, then continue the original images."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import datetime,json,subprocess,sys,time,traceback
from pathlib import Path
vp=Path(sys.argv[1]);v=json.loads(vp.read_bytes());sp=Path(v['study_policy']['path']);q=json.loads(sp.read_bytes());sys.path.insert(0,q['code_root'])
import atm_error_geometry_contract_20261104_v1 as K
parent,ctx,e,C=K.setup(q);K.checked(v['study_policy']);K.checked(v['original_CPU_failure']);K.checked(v['original_controller_failure'])
assert K.sha(__file__)==v['sources']['controller']['sha256'];out=Path(q['output_root']);dest=Path(v['output_root']);repo=Path(ctx['base']['ctx']['relative']['repository_root'])
import brainhive_owned_process_CPU_20261029_v1 as P
until=time.monotonic()+10
while not (vp.parent/'actual_launch.json').exists():assert time.monotonic()<until;time.sleep(.05)
launch=K.read(vp.parent/'actual_launch.json');assert launch['PID']==os.getpid() and K.process(os.getpid())['start_ticks']==launch['start_ticks']
deadline=v['absolute_deadline_epoch'];assert launch['absolute_deadline_epoch']==deadline;started=time.monotonic();owner=child=None;events=[]
def execute(command,label,gpu=False):
    global owner,child
    assert time.time()+930<deadline;K.resources(q,parent,ctx,e,C,starting=True)
    receipt=repo/'audit'/(label+'_job.json')
    if gpu:
        assert not receipt.exists() and not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        command=[sys.executable,'-B',str(repo/'run_gpu_job_v2.py'),'--name',label,'--max-seconds','900','--',*command]
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0' if gpu else '';env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    K.write(dest/'progress.json',dict(phase=label,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    with (dest/(label+'.log')).open('xb') as log:
        owner=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        child=P.capture_wrapper_child(owner,receipt) if gpu else None;rc=owner.wait(timeout=930)
    assert rc==0,dict(label=label,exit_code=rc)
    if gpu:
        job=K.read(receipt);assert job['status']=='COMPLETED' and job['exit_code']==0 and not K.process(job['pid'])['live'] and not K.process(job['wrapper_pid'])['live']
    events.append(dict(label=label,GPU=gpu,receipt=K.ref(receipt) if gpu else None,exit_code=rc));owner=child=None
def stage(source,label,arguments=(),gpu=False):
    execute([sys.executable,'-B',str(Path(q['code_root'])/q['programs'][source]),'--policy',str(sp),*arguments],label,gpu)
def finish(n):
    stage('audit_score',f'audit_images{n}',['--stage',str(n),'--mode','audit'])
    stage('encode',q['job_prefix']+f'_features{n}',['--stage',str(n)],True)
    stage('audit_score',f'score_images{n}',['--stage',str(n),'--mode','score'])
try:
    execute([sys.executable,'-B',v['sources']['verify']['path'],'--study-policy',str(sp),'--verification-policy',str(vp)],v['job_name'],True)
    proof=K.read(dest/'completion.json');assert proof['status']=='PASS_ALL400_ALL50_NATIVE_COORDINATE_REPLAY_ORIGINAL_RAW_V_FAILURE_RETAINED'
    for arm in K.NEW_ARMS:
        stage('generate',q['job_prefix']+f'_gen80_{arm}_0_0',['--stage','80','--arm',arm,'--start','0','--part','0'],True)
    finish(80);action=K.read(out/'readout80.json')['interpretation']['action'];assert action in ('STOP_ALL_NEW_ARMS_JOINT_CLEAR_HARM','EXPAND_ALL_NEW_ARMS_TO400')
    if action=='EXPAND_ALL_NEW_ARMS_TO400':
        for arm in K.NEW_ARMS:
            for begin in (80,160,240,320):
                label=q['job_prefix']+f'_gen400_{arm}_{begin}_0'
                stage('generate',label,['--stage','400','--arm',arm,'--start',str(begin),'--part','0'],True)
                assert K.read(out/(label+'_completion.json'))['status']=='COMPLETE_FIXED_BLOCK'
        finish(400);final=out/'readout400.json'
    else:final=out/'readout80.json'
    K.write(out/'completion.json',dict(status='COMPLETE_NATIVE_ATM_EEG_ERROR_GEOMETRY_PRIORS_PAIRED_IMAGES',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),policy=K.ref(sp),
        final_readout=K.ref(final),initial80_action=action,stage400_completed=final.name=='readout400.json',prepared=K.ref(out/'prepared.json'),events=events,
        seconds=time.monotonic()-started,absolute_deadline_epoch=deadline,new_training_updates=7800,new_models=4,second_innovation_or_SOTA=False,
        engineering_recovery=q['engineering_recovery'],original_failure_preserved=q['original_failure'],native_coordinate_verification=K.ref(dest/'completion.json'),
        verification_policy=K.ref(vp),original_raw_v_gate_NOT_claimed_passed=True,original_CPU_failure_preserved=v['original_CPU_failure']))
    K.write(dest/'controller_completion.json',dict(status='COMPLETE_IMAGE_CONTINUATION_WITH_FROZEN_NATIVE_VERIFICATION',completion=K.ref(out/'completion.json'),events=events))
except BaseException as exc:
    cleanup=P.stop_owned(owner,child);K.write(dest/'controller_failure.json',dict(status='FAILED_PRESERVED_NO_AUTOMATIC_RETRY',error=repr(exc),traceback=traceback.format_exc(),events=events,cleanup=cleanup,absolute_deadline_epoch=deadline));raise
