"""One fixed complete T diagnostic, no metric-based routing or new training."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import datetime,subprocess,sys,time,traceback
from pathlib import Path
import atm_true_forward_noise_contract_20261103_v1 as F
path=Path(sys.argv[1]);c=F.setup(path);s=c['s'];K=c['K']
import brainhive_owned_process_CPU_20261029_v1 as P
code=Path(s['code_root']);root=Path(s['output_root']);repo=Path(c['ctx']['relative']['repository_root'])
until=time.monotonic()+10
while not (code/'actual_launch_v2.json').exists():
    assert time.monotonic()<until;time.sleep(.05)
launch=K.read(code/'actual_launch_v2.json');assert launch['PID']==os.getpid() and K.process(os.getpid())['start_ticks']==launch['start_ticks']
deadline=launch['absolute_deadline_epoch'];started=time.monotonic();owner=child=None;events=[]
try:
    for gpu,mode in ((True,'GPU'),(False,'CPU')):
        assert time.time()+930<deadline
        F.resources(c,starting=True)
        command=[sys.executable,'-B',str(code/s['worker']),'--policy',str(path),'--mode',mode]
        receipt=repo/'audit'/(s['job_name']+'_job.json')
        if gpu:
            assert not receipt.exists()
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
            command=[sys.executable,'-B',str(repo/'run_gpu_job_v2.py'),'--name',s['job_name'],'--max-seconds','900','--',*command]
        K.atomic(root/'controller_progress.json',dict(phase=mode,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0' if gpu else '';env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        with (root/(mode+'_continuation_v2.log')).open('xb') as log:
            owner=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            child=P.capture_wrapper_child(owner,receipt) if gpu else None
            rc=owner.wait(timeout=930)
        assert rc==0,dict(mode=mode,returncode=rc)
        if gpu:
            job=K.read(receipt);assert job['wrapper_pid']==owner.pid and job['status']=='COMPLETED' and job['exit_code']==0
            assert not K.process(job['pid'])['live'] and not K.process(job['wrapper_pid'])['live']
        events.append(dict(phase=mode,GPU=gpu,returncode=rc,receipt=K.ref(receipt) if gpu else None))
        owner=child=None
    result=K.read(root/'readout.json');assert result['status']=='COMPLETE_FULL_T_TRUE_FORWARD_NATIVE_EEG_SCORE_DIAGNOSTIC'
    K.atomic(root/'completion.json',dict(status='COMPLETE_FROZEN_ALL_T_TRUE_FORWARD_NATIVE_EEG_SCORE_DIAGNOSTIC',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),policy=K.ref(path),events=events,
        readout=K.ref(root/'readout.json'),seconds=time.monotonic()-started,absolute_deadline_epoch=deadline,
        new_training_updates=0,new_images=0,new_independent_second_method_or_SOTA=False,
        engineering_continuation=dict(original_policy=s['original_policy'],original_failure=s['original_failure'],
            original_deadline_unchanged=deadline,completed_role=s['completed_role'],
            preserved_role_metric_SHA256=s['preserved_role_metric_SHA256'],preserved_role_risk_SHA256=s['preserved_role_risk_SHA256'],
            only_incomplete_second_role_new_GPU=True,metadata_copy_to_CPU_only=True)))
except BaseException as exc:
    cleanup=P.stop_owned(owner,child)
    K.atomic(root/'engineering_continuation_failure_v2.json',dict(status='FAILED_PREFIX_RETAINED_NO_AUTOMATIC_SCIENTIFIC_RETRY',error=repr(exc),
        traceback=traceback.format_exc(),events=events,owned_cleanup=cleanup,absolute_deadline_epoch=deadline));raise
