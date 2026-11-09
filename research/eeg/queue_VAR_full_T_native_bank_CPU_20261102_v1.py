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
import VAR_full_T_native_bank_contract_CPU_20261102_v1 as C


def run(protocol):
    q=C.setup(protocol);c,n,prev,gate=C.dependency(q);C.resources(q,starting=True)
    m=n.mean_contract();parent=C.read(m.PARENT['path']);repo=Path(parent['repository_root'])
    sys.path.insert(0,str(Path(m.PARENT['path']).parent))
    from brainhive_gpu_exit_contract_CPU_20261101_v1 import verify
    spec=importlib.util.spec_from_file_location('owned_VAR_batch64_top0_wrapper',m.checked(m.OWNED))
    owned=importlib.util.module_from_spec(spec);spec.loader.exec_module(owned)
    assert not (C.OUT/'queue_started.json').exists()
    C.write(C.OUT/'queue_started.json',dict(PID=os.getpid(),protocol=C.ref(protocol),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),gate=gate))
    started=time.monotonic();owner=pinned=None;events=[];phase='START'
    try:
        tasks=[(f'part_{k:03d}',C.CODE/'build_VAR_full_T_native_bank_GPU_20261102_v1.py',True,k) for k in range(207)]
        tasks.append(('whole_bank_CPU',C.CODE/'audit_VAR_full_T_native_bank_CPU_20261102_v1.py',False,None))
        for label,entry,gpu,index in tasks:
            phase=label;C.resources(q,starting=gpu);name='VAR_full_T_native_bank_20261102_v1_'+label
            command=[sys.executable,'-B',str(entry),'--protocol',str(protocol)]
            if index is not None:command+=['--part',str(index)]
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
                C.write(C.OUT/'progress.json',dict(phase=label,completed_parts=len([e for e in events if e['GPU']]),total_parts=207,active_PID=owner.pid,active_start_ticks=identity['start_ticks'],UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()),update=True)
                while owner.poll() is None:
                    assert time.monotonic()-tick<(965 if gpu else q['CPU_seconds'])
                    assert time.monotonic()-started<q['maximum_execution_seconds'];time.sleep(2)
                rc=owner.wait()
            event=dict(label=label,exit_code=rc,seconds=time.monotonic()-tick,GPU=gpu)
            if gpu:event.update(verify(rc,C.read(repo/'audit'/(name+'_job.json')),owner.pid));event['GPU_receipt']=C.ref(repo/'audit'/(name+'_job.json'))
            events.append(event);C.write(C.OUT/(label+'_event.json'),event);owner=pinned=None;assert rc==0
        C.write(C.OUT/'completion.json',dict(status='COMPLETE_FULL_T13230_NATIVE_K1000_ALL680_BANK_AND_WHOLE_CPU_AUDIT',protocol=C.ref(protocol),CPU_audit=C.ref(C.OUT/'whole_bank_CPU_audit.json'),events=events,seconds=time.monotonic()-started,new_optimizer_updates=0,new_research_images=0,EEG_fit_admitted=False,NEW_contacted=False))
        C.resources(q)
    except BaseException as error:
        if owner is not None:owned.stop_owned(owner,pinned)
        C.write(C.OUT/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),events=events,no_automatic_retry=True));raise


if __name__=='__main__':
    def stop(*unused):raise InterruptedError('Stop only this owned execution engineering controller')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);run(p.parse_args().protocol)
