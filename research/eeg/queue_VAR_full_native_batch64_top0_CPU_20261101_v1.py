"""One finite GPU engineering job and its whole CPU audit, no scientific retry."""
import argparse
import datetime
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import VAR_full_native_throughput_contract_CPU_20261101_v1 as C


def run(protocol):
    q=C.setup(protocol);n,gate=C.dependency(q);C.resources(q,starting=True)
    m=n.mean_contract();parent=C.read(m.PARENT['path']);repo=Path(parent['repository_root'])
    sys.path.insert(0,str(Path(m.PARENT['path']).parent))
    from brainhive_gpu_exit_contract_CPU_20261101_v1 import verify
    spec=importlib.util.spec_from_file_location('owned_VAR_batch64_top0_wrapper',m.checked(m.OWNED))
    owned=importlib.util.module_from_spec(spec);spec.loader.exec_module(owned)
    assert not (C.OUT/'queue_started.json').exists()
    C.write(C.OUT/'queue_started.json',dict(PID=os.getpid(),protocol=C.ref(protocol),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),gate=gate))
    started=time.monotonic();owner=pinned=None;events=[];phase='START'
    try:
        for label,entry,gpu in [('batch64_full_T8_and_top0',C.CODE/'profile_VAR_full_native_batch64_top0_GPU_20261101_v1.py',True),
                               ('whole_CPU_audit',C.CODE/'audit_VAR_full_native_batch64_top0_CPU_20261101_v1.py',False)]:
            phase=label;C.resources(q,starting=gpu);name='VAR_native_batch64_top0_20261101_v1_'+label
            command=[sys.executable,'-B',str(entry),'--protocol',str(protocol)]
            if gpu:
                assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
                command=[sys.executable,'-B',str(repo/'run_gpu_job_v2.py'),'--name',name,'--max-seconds','900','--']+command+['--job-name',name]
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0' if gpu else '',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            tick=time.monotonic()
            with (C.OUT/(label+'.stdout.log')).open('xb') as stdout,(C.OUT/(label+'.stderr.log')).open('xb') as stderr:
                owner=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
                pinned=owned.capture_wrapper_child(owner,repo/'audit'/(name+'_job.json')) if gpu else None
                identity=C.process(owner.pid,-1)
                C.write(C.OUT/(label+'_identity.json'),dict(PID=owner.pid,start_ticks=identity['start_ticks'],job_name=name,GPU=gpu))
                while owner.poll() is None:
                    assert time.monotonic()-tick<(965 if gpu else q['CPU_seconds'])
                    assert time.monotonic()-started<q['maximum_execution_seconds'];time.sleep(2)
                rc=owner.wait()
            event=dict(label=label,exit_code=rc,seconds=time.monotonic()-tick,GPU=gpu)
            if gpu:event.update(verify(rc,C.read(repo/'audit'/(name+'_job.json')),owner.pid));event['GPU_receipt']=C.ref(repo/'audit'/(name+'_job.json'))
            events.append(event);C.write(C.OUT/(label+'_event.json'),event);owner=pinned=None;assert rc==0
        C.write(C.OUT/'completion.json',dict(status='COMPLETE_FULL_NATIVE_FP32_BATCH64_AND_MATCHING_TOP0_ENGINEERING_ONLY',
            protocol=C.ref(protocol),profile=C.ref(C.OUT/'profile/completion.json'),CPU_audit=C.ref(C.OUT/'independent_CPU_audit.json'),
            events=events,seconds=time.monotonic()-started,new_optimizer_updates=0,new_research_images=0,
            no_full_T_bank_or_EEG_fit_admitted=True,NEW_contacted=False))
        C.resources(q)
    except BaseException as error:
        if owner is not None:owned.stop_owned(owner,pinned)
        C.write(C.OUT/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),events=events,no_automatic_retry=True));raise


if __name__=='__main__':
    def stop(*unused):raise InterruptedError('Stop only this owned execution engineering controller')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);run(p.parse_args().protocol)
