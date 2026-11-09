"""One bounded physical-measurement engineering job on OLD; no automatic retry."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import datetime, subprocess, sys, time, traceback
from pathlib import Path
qpath = Path(sys.argv[1]); import json
q = json.loads(qpath.read_bytes()); sys.path.insert(0, str(Path(q['reference_policy']['path']).parent))
import enigma_native_reference_contract_20261104_v1 as C
c = C.setup(q['reference_policy']['path']); K = c['K']; repo = c['repo']; out = Path(q['output_root'])
assert K.ref(__file__) == q['sources']['controller']
import brainhive_owned_process_CPU_20261029_v1 as P
started=time.monotonic(); owner=child=None
until=time.monotonic()+10
while not (qpath.parent/'actual_launch.json').exists(): assert time.monotonic()<until;time.sleep(.05)
launch=K.read(qpath.parent/'actual_launch.json');assert launch['PID']==os.getpid()
assert K.process(os.getpid())['start_ticks']==launch['start_ticks']
label=q['job_prefix']+'_measure'; receipt=repo/'audit'/(label+'_job.json')
try:
    before=C.resources(c,starting=True)
    assert before['free_bytes']>=before['required_including_this_project']+q['maximum_output_bytes']
    assert time.time()+930<launch['absolute_deadline_epoch'] and not receipt.exists()
    command=[sys.executable,'-B',str(repo/'run_gpu_job_v2.py'),'--name',label,'--max-seconds','900','--',
             sys.executable,'-B',q['sources']['GPU']['path'],'--policy',str(qpath)]
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0';env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    with (out/'measure.log').open('xb') as log:
        owner=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        child=P.capture_wrapper_child(owner,receipt);rc=owner.wait(timeout=930)
    job=K.read(receipt);assert rc==0 and job['status']=='COMPLETED' and job['exit_code']==0
    assert not K.process(job['pid'])['live'] and not K.process(job['wrapper_pid'])['live']
    K.write(out/'controller_completion.json',dict(status='COMPLETE_SINGLE_MEASUREMENT_ENGINEERING_JOB',policy=K.ref(qpath),
        receipt=K.ref(receipt), scientific_completion=K.ref(out/'completion.json'),seconds=time.monotonic()-started,
        absolute_deadline_epoch=launch['absolute_deadline_epoch'], new_updates=0,new_images=0))
except BaseException as exc:
    cleanup=P.stop_owned(owner,child)
    K.write(out/'controller_failure.json',dict(status='FAILED_PREFIX_RETAINED_NO_AUTO_RETRY',error=repr(exc),
        traceback=traceback.format_exc(),cleanup=cleanup,seconds=time.monotonic()-started));raise
