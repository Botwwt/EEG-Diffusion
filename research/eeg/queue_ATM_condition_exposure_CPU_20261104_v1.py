"""One finite diagnostic, with exact owned-process cleanup and no retry."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import datetime,subprocess,sys,time,traceback
from pathlib import Path
import atm_condition_exposure_contract_20261104_v1 as X
path=Path(sys.argv[1]);c=X.setup(path);s=c['s'];K=c['K'];root=Path(s['output_root']);code=Path(s['code_root']);repo=Path(c['base']['ctx']['relative']['repository_root'])
import brainhive_owned_process_CPU_20261029_v1 as P
until=time.monotonic()+10
while not (code/'actual_launch.json').exists():assert time.monotonic()<until;time.sleep(.05)
launch=K.read(code/'actual_launch.json');assert launch['PID']==os.getpid() and K.process(os.getpid())['start_ticks']==launch['start_ticks']
deadline=launch['absolute_deadline_epoch'];owner=child=None;events=[];started=time.monotonic()
try:
    for mode in ('GPU','CPU'):
        assert time.time()+930<deadline;X.resources(c,starting=True)
        command=[sys.executable,'-B',str(code/s['worker']),'--policy',str(path),'--mode',mode]
        receipt=repo/'audit'/(s['job_name']+'_job.json');gpu=mode=='GPU'
        if gpu:
            assert not receipt.exists()
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
            command=[sys.executable,'-B',str(repo/'run_gpu_job_v2.py'),'--name',s['job_name'],'--max-seconds','900','--',*command]
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0' if gpu else ''
        K.atomic(root/'controller_progress.json',dict(phase=mode,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        with (root/(mode+'.log')).open('xb') as log:
            owner=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            child=P.capture_wrapper_child(owner,receipt) if gpu else None;rc=owner.wait(timeout=930)
        assert rc==0,dict(mode=mode,returncode=rc)
        if gpu:
            job=K.read(receipt);assert job['wrapper_pid']==owner.pid and job['status']=='COMPLETED' and job['exit_code']==0
            assert not K.process(job['pid'])['live'] and not K.process(job['wrapper_pid'])['live']
        events.append(dict(mode=mode,returncode=rc,receipt=K.ref(receipt) if gpu else None));owner=child=None
    assert K.read(root/'readout.json')['status']=='COMPLETE_FULL_T_FROZEN_CONDITION_EXPOSURE_ROLE_DIAGNOSTIC'
    K.atomic(root/'completion.json',dict(status='COMPLETE_FROZEN_CONDITION_EXPOSURE_ROLE_ALL_T',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        policy=K.ref(path),readout=K.ref(root/'readout.json'),events=events,seconds=time.monotonic()-started,absolute_deadline_epoch=deadline,
        new_training_updates=0,new_images=0,no_new_method_or_inference_gain=True))
except BaseException as exc:
    cleanup=P.stop_owned(owner,child);K.atomic(root/'failure.json',dict(status='FAILED_PREFIX_RETAINED_NO_RETRY',error=repr(exc),traceback=traceback.format_exc(),events=events,cleanup=cleanup,absolute_deadline_epoch=deadline));raise
