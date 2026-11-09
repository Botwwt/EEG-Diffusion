"""One finite native profile after exact original/mean/control dependencies."""
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
import VAR_multiscale_profile_contract_CPU_20261101_v1 as C


def progress(value):
    p=C.OUT/('progress.pending.'+str(os.getpid()))
    with p.open('x') as stream:__import__('json').dump(value,stream,indent=2);stream.write('\n')
    os.replace(p,C.OUT/'progress.json')


def run(protocol):
    s=C.setup(protocol,require_ready=False)
    assert C.OUT.exists() and not (C.OUT/'queue_started.json').exists()
    C.write(C.OUT/'queue_started.json',dict(PID=os.getpid(),protocol=C.ref(protocol),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    M=C.mean_contract();existing=C.read(C.checked(M.PARENT));R=Path(existing['repository_root'])
    sys.path.insert(0,str(Path(M.PARENT['path']).parent))
    from brainhive_gpu_exit_contract_CPU_20261101_v1 import verify
    spec=importlib.util.spec_from_file_location('VAR_profile_owned_wrapper',M.checked(M.OWNED))
    owned=importlib.util.module_from_spec(spec);spec.loader.exec_module(owned)
    logs=C.OUT/'queue';logs.mkdir();owner=pinned=None;events=[];phase='WAIT';started=None
    try:
        last=None;last_saved=0.
        while True:
            assert time.time()<=s['waiting_deadline_epoch'],'Finite conditional waiting deadline exhausted'
            gate=C.dependency(s);phase=gate['status']
            if phase.startswith('STOP_'):raise RuntimeError(phase)
            if phase=='READY_ALL_THREE_COMPLETE_AND_EXITED':
                try:available=C.resources(s,starting=True)
                except AssertionError:phase='WAIT_RESOURCES_AFTER_THREE_SUCCESSFUL_QUEUES'
                else:
                    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
                    if not apps:
                        C.write(C.OUT/'actual_dependency_closure.json',dict(**gate,resources=available,GPU_empty=True));break
                    phase='WAIT_OTHER_GPU_USER_AFTER_THREE_SUCCESSFUL_QUEUES'
            if phase!=last or time.monotonic()-last_saved>300:
                progress(dict(PID=os.getpid(),phase=phase,dependency=gate,new_GPU_jobs=0,new_optimizer_updates=0,
                    UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()));last=phase;last_saved=time.monotonic()
            time.sleep(15)
        started=time.monotonic()
        for label,args,gpu in [
            ('native_T8_K1000_680_profile',[C.CODE/'profile_VAR_multiscale_native_GPU_20261101_v1.py','--protocol',protocol],True),
            ('independent_full_cell_CPU_audit',[C.CODE/'audit_VAR_multiscale_native_profile_CPU_20261101_v1.py','--protocol',protocol],False)]:
            phase=label;C.resources(s,starting=gpu);name='VAR_multiscale_native_profile_20261101_v1_'+label
            command=[sys.executable,'-B']+list(map(str,args))
            if gpu:
                assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
                command=[sys.executable,'-B',str(R/'run_gpu_job_v2.py'),'--name',name,'--max-seconds','900','--']+command+['--job-name',name]
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0' if gpu else '',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            tick=time.monotonic()
            with (logs/(label+'.stdout.log')).open('xb') as stdout,(logs/(label+'.stderr.log')).open('xb') as stderr:
                owner=subprocess.Popen(command,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
                pinned=owned.capture_wrapper_child(owner,R/'audit'/(name+'_job.json')) if gpu else None
                ident=M.process(owner.pid,-1)
                progress(dict(PID=os.getpid(),phase=phase,active_PID=owner.pid,active_start_ticks=ident['start_ticks'],GPU=gpu,job_name=name,
                    UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
                while owner.poll() is None:
                    assert time.monotonic()-tick<(965 if gpu else 600) and time.monotonic()-started<s['execution_seconds']
                    time.sleep(2)
                rc=owner.wait()
            event=dict(label=label,exit_code=rc,seconds=time.monotonic()-tick,GPU=gpu)
            if gpu:
                receipt=C.read(R/'audit'/(name+'_job.json'));event.update(verify(rc,receipt,owner.pid));event['GPU_receipt']=C.ref(R/'audit'/(name+'_job.json'))
            events.append(event);C.write(logs/(label+'.json'),event);owner=pinned=None;assert rc==0
        assert C.read(C.OUT/'actual_independent_CPU_audit.json')['status']=='PASS_ALL80000_NATIVE_SCALE_SUMS_AND_FIXED_ENGINEERING_ARTIFACTS_CPU'
        C.write(C.OUT/'completion.json',dict(status='COMPLETE_T8_K1000_680_NATIVE_ENGINEERING_AND_INDEPENDENT_CPU_ONLY',protocol=C.ref(protocol),
            profile=C.ref(C.OUT/'profile/completion.json'),CPU_audit=C.ref(C.OUT/'actual_independent_CPU_audit.json'),events=events,
            seconds=time.monotonic()-started,new_research_images=0,new_optimizer_updates=0,
            full_T_emission_bank_or_EEG_fit_admitted=False,second_method_or_SOTA_proved=False,NEW_contacted=False))
    except BaseException as error:
        if owner is not None:owned.stop_owned(owner,pinned)
        C.write(C.OUT/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),events=events,no_automatic_retry=True));raise


if __name__=='__main__':
    def stop(*unused):raise InterruptedError('Stop only this owned native VAR profile controller')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);run(p.parse_args().protocol)
