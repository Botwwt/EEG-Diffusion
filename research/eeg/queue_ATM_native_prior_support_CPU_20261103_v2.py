"""One GPU diagnostic then a distinct CPU audit, with bounded exact ownership."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import datetime
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

path=Path(sys.argv[1]);q=json.loads(path.read_bytes())
sys.path.insert(0,str(Path(q['base_policy']['path']).parent))
import brainhive_relative_fixed_output_contract_20261103_v1 as K
import brainhive_owned_process_CPU_20261029_v1 as P
ctx=K.setup(str(K.checked(q['base_policy'])))
code=Path(q['code_root']);root=Path(q['output_root']);repo=Path(ctx['relative']['repository_root'])
until=time.monotonic()+10
while not (code/'actual_launch_v2.json').exists():
    assert time.monotonic()<until
    time.sleep(.05)
launch=K.read(code/'actual_launch_v2.json');deadline=launch['absolute_deadline_epoch']
assert launch['PID']==os.getpid() and K.process(os.getpid())['start_ticks']==launch['start_ticks']
owner=child=None;events=[];started=time.monotonic()
try:
    for gpu,mode in ((True,'sample'),(False,'audit')):
        assert time.time()+930<deadline
        command=[sys.executable,'-B',str(code/q['worker']),'--policy',str(path),'--mode',mode]
        receipt=repo/'audit'/(q['job_name']+'_job.json')
        if gpu:
            assert not receipt.exists()
            command=[sys.executable,'-B',str(repo/'run_gpu_job_v2.py'),'--name',q['job_name'],'--max-seconds','900','--',*command]
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0' if gpu else '';env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        K.atomic(root/'controller_progress.json',dict(phase=mode,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        with (root/(mode+'_trace_v2.log')).open('xb') as log:
            owner=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            child=P.capture_wrapper_child(owner,receipt) if gpu else None
            rc=owner.wait(timeout=930)
        assert rc==0,dict(phase=mode,returncode=rc)
        if gpu:
            job=K.read(receipt);assert job['wrapper_pid']==owner.pid and job['status']=='COMPLETED' and job['exit_code']==0
            assert not K.process(job['pid'])['live'] and not K.process(job['wrapper_pid'])['live']
        events.append(dict(phase=mode,returncode=rc,GPU=gpu,receipt=K.ref(receipt) if gpu else None))
        owner=child=None
    readout=K.read(root/'readout.json');assert readout['status']=='COMPLETE_ALL_T_FROZEN_PRIOR_CLIP_CONFIGURATION_DIAGNOSTIC'
    K.atomic(root/'completion.json',dict(status='COMPLETE_ONE_NATIVE_PRIOR_T_ONLY_CONFIG_DIAGNOSTIC',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),policy=K.ref(path),readout=K.ref(root/'readout.json'),
        events=events,absolute_deadline_epoch=deadline,seconds=time.monotonic()-started,new_training_updates=0,new_images=0,not_new_second_innovation=True,
        engineering_recovery=dict(original_policy=q['original_policy'],original_failure=q['original_failure'],
            preserved_first128=q['preserved_first128'],original_deadline_epoch=q['original_deadline_epoch'],
            newly_registered_engineering_deadline_epoch=deadline,original_window_not_claimed_passed=True,
            same_batch128_trace_only=True,no_weights_data_seed_or_scientific_config_change=True)))
except BaseException as exc:
    cleanup=P.stop_owned(owner,child)
    K.atomic(root/'trace_recovery_failure_v2.json',dict(status='FAILED_PREFIX_RETAINED_NO_AUTOMATIC_SCIENTIFIC_RETRY',
        error=repr(exc),traceback=traceback.format_exc(),events=events,owned_cleanup=cleanup,absolute_deadline_epoch=deadline))
    raise
