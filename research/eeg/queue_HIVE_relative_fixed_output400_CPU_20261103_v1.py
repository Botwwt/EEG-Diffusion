"""One complete frozen four-arm diagnostic, with no score-based branching."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import datetime
import subprocess
import sys
import time
import traceback
from pathlib import Path

import brainhive_relative_fixed_output_contract_20261103_v1 as K
import brainhive_owned_process_CPU_20261029_v1 as P

path=Path(sys.argv[1]);ctx=K.setup(path);s=ctx['s'];root=Path(s['output_root'])
code=Path(s['code_root']);repo=Path(ctx['relative']['repository_root'])
launch_path=code/'actual_launch.json'
launch_wait=time.monotonic()+10
while not launch_path.exists():
    assert time.monotonic()<launch_wait,'The new owner launch receipt was not committed'
    time.sleep(.05)
launch=K.read(launch_path);deadline=launch['absolute_deadline_epoch']
assert not (root/'completion.json').exists() and not (root/'failure.json').exists()
assert launch['PID']==os.getpid() and K.process(os.getpid())['start_ticks']==launch['start_ticks']
started=time.monotonic();events=[];child=None;owner=None


def stage(label,source,args,gpu):
    global child,owner
    assert time.time()+930<deadline
    resources=K.resources(ctx,starting=True)
    if gpu:
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    K.atomic(root/'progress.json',dict(phase=label,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        completed_events=events,full400_effects_read=False))
    command=[sys.executable,'-B',str(code/source),'--policy',str(path),*args]
    name=s['job_prefix']+'_'+label;receipt=repo/'audit'/(name+'_job.json')
    if gpu:
        assert not receipt.exists()
        command=[sys.executable,'-B',str(repo/'run_gpu_job_v2.py'),'--name',name,'--max-seconds','900','--',*command]
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0' if gpu else ''
    env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    with (root/(label+'.log')).open('xb') as log:
        owner=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        child=P.capture_wrapper_child(owner,receipt) if gpu else None
        returncode=owner.wait(timeout=930)
    if gpu:
        job=K.read(receipt)
        assert job['wrapper_pid']==owner.pid and job['status']=='COMPLETED' and job['exit_code']==0
        assert not K.process(job['pid'])['live'] and not K.process(job['wrapper_pid'])['live']
    assert returncode==0,dict(label=label,returncode=returncode)
    events.append(dict(label=label,GPU=gpu,owner_PID=owner.pid,returncode=returncode,
        wrapper_receipt=K.ref(receipt) if gpu else None,seconds_since_start=time.monotonic()-started))
    child=owner=None


try:
    preflight=K.read(root/'CPU_preflight/completion.json')
    assert preflight['status']=='PASS_REAL_FROZEN4_RAW1_HEADS_ALL400_NUMPY64_AND_BASELINE_IDENTITIES'
    assert preflight['policy']==K.ref(path)
    for lo in (0,80,160,240,320):
        stage(f'images_{lo:03d}',s['filenames']['GPU'],['--mode','generate','--lo',str(lo),'--hi',str(lo+80)],True)
        block=K.read(root/f'generate_{lo:03d}_{lo+80:03d}/completion.json')
        assert block['range']==[lo,lo+80] and block['new_images']+block['reused_images']==320
    stage('features400',s['filenames']['GPU'],['--mode','features'],True)
    stage('CPU_all400_audit_and_score',s['filenames']['CPU'],['--mode','score'],False)
    result=K.read(root/'readout400.json')
    assert result['status']=='COMPLETE_FROZEN_RAW1_ALL400_ALL8METHODS_ALL8METRICS'
    assert result['all3200_images_actual_identity_SHA_pixels_noise_audited']
    blocks=[K.read(root/f'generate_{lo:03d}_{lo+80:03d}/completion.json') for lo in (0,80,160,240,320)]
    K.atomic(root/'completion.json',dict(status='COMPLETE_ONE_FROZEN_RELATIVE_RAW1_FULL400_DIAGNOSTIC',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),policy=K.ref(path),readout=K.ref(root/'readout400.json'),
        events=events,queries=400,new_images=sum(v['new_images'] for v in blocks),
        identical_existing_images_reused=sum(v['reused_images'] for v in blocks),
        all1600_conditioned_images=True,baseline_images=1600,new_training_updates=0,
        old_calibrated_method_STOP_unchanged=True,not_new_second_method_or_SOTA=True,
        original_deadline_epoch=deadline,seconds=time.monotonic()-started,resources=K.resources(ctx)))
    K.atomic(root/'progress.json',dict(phase='COMPLETE_ALL400_NO_AUTOMATIC_FURTHER_FIT'))
except BaseException as exc:
    cleanup=P.stop_owned(owner,child)
    K.atomic(root/'failure.json',dict(status='FAILED_FROZEN_DIAGNOSTIC_ALL_PREFIXES_RETAINED_NO_AUTOMATIC_RETRY',
        error=repr(exc),traceback=traceback.format_exc(),owned_cleanup=cleanup,events=events,
        original_deadline_epoch=deadline,seconds=time.monotonic()-started));raise
