"""Finite CPU waiter, never interrupts original experiment or uses reserved NEW."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import datetime, importlib.util, json, selectors, subprocess, sys, time, traceback
from pathlib import Path

path=Path(sys.argv[1]);p=json.loads(path.read_bytes());root=path.parent
sys.path[:0]=[str(root),p['repo'],str(Path(p['original_policy']).parent)]
import atm_continuous_margin_contract_20261106_v204 as C
c=C.setup(p['original_policy']);q,K=c['q'],c['K'];launch=K.read(Path(p['original_policy']).parent/'actual_launch.json')
assert (launch['PID'],launch['start_ticks'])==(780584,156317640)
spec=importlib.util.spec_from_file_location('owned_gpu_profile',K.checked(c['physical']['owned_process']))
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
child=gpu_child=None
def resources(starting=False):
    r=C.resources(c,starting)
    used=K.allocated(root);assert used<=p['maximum_output_bytes']
    assert r['free_bytes']>=r['required_including_this_project']+p['maximum_output_bytes']+p['wrapper_reserve_bytes']-used
    return r
def status(phase,**kw):
    resources();K.write(root/'progress.json',dict(phase=phase,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw))
try:
    for f,pin in p['sources'].items():K.checked(pin)
    status('WAIT_ORIGINAL_FULL_STUDY_AND_TERMINAL_REVIEW',GPU_jobs_started=0)
    while True:
        assert time.time()<p['absolute_wait_deadline_epoch'],'finite wait expired; no GPU launched'
        if (c['out']/'failure.json').exists():
            raise RuntimeError('original study failed; preserve it and do not start benchmark')
        owner=K.process(launch['PID'],launch['start_ticks'])
        done=K.read(c['out']/'completion.json') if (c['out']/'completion.json').exists() else {}
        rp=Path(q['project_root'])/'terminal_review/completion.json'
        review=K.read(rp) if rp.exists() else {}
        if not owner['live'] and done.get('status')=='COMPLETE_CONTINUOUS_MARGIN_TRAINING_AND_IMAGES' and review.get('status')=='PASS_ALL2400_IDENTITIES_PIXELS_NOISES_11200_RANK_VALUES_7200_LOWLEVEL_138_CIS':
            break
        resources();time.sleep(60)
    resources(True)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    command=[sys.executable,'-B',c['physical']['GPU_wrapper']['path'],'--name',p['job_name'],'--max-seconds','850','--',
             sys.executable,'-B',p['sources']['profile']['path'],'--policy',str(path)]
    status('ORIGINAL_CLOSED_START_ONE_GPU_MICROBENCHMARK',command=command)
    child=subprocess.Popen(command,cwd=c['repo'],env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),
                           stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
    identity=K.process(child.pid);K.write(root/'GPU_wrapper_launch.json',dict(identity=identity,command=command))
    receipt=c['repo']/'audit'/(p['job_name']+'_job.json')
    gpu_child=P.capture_wrapper_child(child,receipt)
    selector=selectors.DefaultSelector();selector.register(child.stdout,selectors.EVENT_READ)
    started=time.monotonic();written=0
    with (root/'GPU_worker.log').open('xb') as log:
        while selector.get_map():
            assert time.monotonic()-started<885
            for key,_ in selector.select(timeout=1):
                chunk=os.read(key.fd,8192)
                if not chunk:selector.unregister(key.fileobj);continue
                written+=len(chunk);assert written<=128<<10,'log cap reached; failure preserved'
                log.write(chunk);log.flush()
        rc=child.wait(timeout=10)
    assert rc==0,rc
    assert K.read(root/'completion.json')['status']=='COMPLETE_REAL_TWO_ENCODER_GPU_MICROBENCHMARK_ONLY'
    status('COMPLETE_TWO_HOST_MICROBENCHMARK_ONLY',GPU_jobs_started=1)
except BaseException as exc:
    cleanup=P.stop_owned(child,gpu_child)
    K.write(root/'queue_failure.json',dict(status='FAILED_PRESERVED_NO_AUTOMATIC_RETRY',error=repr(exc),traceback=traceback.format_exc(),cleanup=cleanup))
    raise
