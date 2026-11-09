import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,subprocess,sys,time,traceback
from pathlib import Path
import atm_repeat_margin_contract_20261105_v1 as C
c=C.setup(sys.argv[1]);q,K,out,pq=c['q'],c['K'],c['out'],c['physical'];assert K.ref(__file__)==q['sources']['controller']
spec=importlib.util.spec_from_file_location('owned',K.checked(pq['owned_process']));P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
started=time.monotonic();owner=child=None;launchpath=Path(q['code_root'])/'actual_launch.json'
while not launchpath.exists():
    assert time.monotonic()-started<15;time.sleep(.05)
launch=K.read(launchpath);assert launch['PID']==os.getpid() and K.process(os.getpid())['start_ticks']==launch['start_ticks']
receipt=Path(pq['GPU_wrapper']['path']).parent/'audit'/(q['job_prefix']+'_measure_job.json')
try:
    C.resources(c,True);assert not receipt.exists();env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
    command=[sys.executable,'-B',pq['GPU_wrapper']['path'],'--name',q['job_prefix']+'_measure','--max-seconds','900','--',sys.executable,'-B',q['sources']['GPU']['path'],sys.argv[1]]
    with (out/'GPU.log').open('xb') as log:
        owner=subprocess.Popen(command,cwd=receipt.parent.parent,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        child=P.capture_wrapper_child(owner,receipt);rc=owner.wait(timeout=min(935,launch['absolute_deadline_epoch']-time.time()))
    job=K.read(receipt);assert rc==0 and job['status']=='COMPLETED' and job['exit_code']==0
    assert not K.process(job['pid'])['live'] and not K.process(job['wrapper_pid'])['live'];owner=child=None;C.resources(c,True)
    with (out/'CPU.log').open('xb') as log:
        owner=subprocess.Popen([sys.executable,'-B',q['sources']['CPU']['path'],sys.argv[1]],cwd=q['code_root'],env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        cpu=K.process(owner.pid);K.write(out/'CPU_owner.json',cpu)
        assert owner.wait(timeout=min(900,launch['absolute_deadline_epoch']-time.time()))==0
    assert not K.process(cpu['PID'],cpu['start_ticks'])['live']
    audit=K.read(out/'independent_CPU_audit.json');assert audit['status']=='PASS_FIXED3200_CPU64_AND_ALL_T_COVERAGE_MASS'
    K.write(out/'completion.json',dict(status='COMPLETE_FROZEN_REPEAT_MARGIN_DIAGNOSTIC',policy=K.ref(sys.argv[1]),scientific=K.ref(out/'scientific_completion.json'),
        CPU_audit=K.ref(out/'independent_CPU_audit.json'),receipt=K.ref(receipt),seconds=time.monotonic()-started,new_updates=0,new_images=0,no_automatic_training=True))
except BaseException as exc:
    cleanup=P.stop_owned(owner,child);K.write(out/'controller_failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),cleanup=cleanup,seconds=time.monotonic()-started));raise
