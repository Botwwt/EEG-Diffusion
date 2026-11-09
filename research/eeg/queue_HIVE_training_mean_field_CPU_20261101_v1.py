"""Finite CPU-only wait/compute/audit queue; never launches a GPU job or a fit."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import brainhive_training_mean_field_CPU_20261101_v1 as M

def headroom():
    host=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    folder=Path('/sys/fs/cgroup')
    maximum=(folder/'memory.max').read_text().strip()
    return min(host,int(maximum)-int((folder/'memory.current').read_text())) if maximum!='max' else host

def used_bytes():
    total=0
    for path in M.OUT.rglob('*'):
        assert not path.is_symlink()
        if path.is_file():total+=path.stat().st_size
    return total

def atomic_progress(value):
    pending=M.OUT/('progress.pending.'+str(os.getpid()))
    with pending.open('w',encoding='utf8') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    os.replace(pending,M.OUT/'progress.json')

def resource_snapshot(s):
    import shutil
    used=used_bytes();assert used<=s['maximum_output_bytes']
    free=shutil.disk_usage(M.OUT).free
    # Keep every parent reservation, even though this queue starts after its
    # completion. No old storage promise is silently reclaimed.
    required=(s['maximum_output_bytes']-used+s['legacy_floor_bytes']+s['disk_buffer_bytes']
              +(14<<30)+(64<<20)+(8<<30))
    assert free>=required
    return dict(effective_CPU_headroom_bytes=headroom(),used_output_bytes=used,
                free_bytes=free,required_remaining_bytes=required,physical_reservation=False)

def run(protocol):
    s=M.validate(M.read(protocol));assert Path(protocol)==M.CODE/'protocol.json'
    for pin in s['source_pins']:M.checked(pin)
    review=M.read(M.CODE/'actual_CPU_only_preparation_review.json')
    assert review['status']=='PASS_SOURCE_BOUND_TRAIN_ONLY_MEAN_CPU_PREPARATION'
    assert review['protocol']==M.ref(protocol) and review['source_pins']==s['source_pins'] and review['GPU_admitted'] is False
    assert M.OUT.exists() and not (M.OUT/'queue_started.json').exists()
    M.write(M.OUT/'queue_started.json',dict(PID=os.getpid(),protocol=M.ref(protocol),
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),new_GPU_jobs=0))
    # Reuse the already independently tested owned-child cleanup implementation.
    sys.path.insert(0,str(Path(M.PARENT['path']).parent))
    parent=M.read(M.PARENT['path']);sys.path.insert(0,parent['repository_root'])
    import importlib.util
    exact_spec=importlib.util.spec_from_file_location('training_mean_owned_process_exact',M.checked(s['owned_process']))
    exact_owned=importlib.util.module_from_spec(exact_spec);exact_spec.loader.exec_module(exact_owned)
    stop_owned=exact_owned.stop_owned
    owner=None;events=[];phase='WAIT';last_state=None;last_saved=0.0
    try:
        while True:
            assert time.time()<=s['waiting_deadline_epoch'],'Original lifecycle absolute waiting deadline exhausted'
            gate=M.dependency(s)
            if gate['status'].startswith('STOP_'):raise RuntimeError(gate['status'])
            current=gate['status']
            if current=='READY_CLOSED_ORIGINAL_LIFECYCLE':
                resources=resource_snapshot(s)
                gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
                if resources['effective_CPU_headroom_bytes']>=s['minimum_start_CPU_headroom_bytes'] and not gpu:
                    M.write(M.OUT/'actual_dependency_closure.json',dict(**gate,resources=resources,GPU_empty=True))
                    break
                current='WAIT_RESOURCES_AFTER_ORIGINAL_CLOSED'
            if current!=last_state or time.monotonic()-last_saved>300:
                atomic_progress(dict(PID=os.getpid(),phase=current,dependency=gate,
                    UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),new_GPU_jobs=0,new_training_updates=0))
                last_state=current;last_saved=time.monotonic()
            time.sleep(15)
        for phase,entry in [('compute','compute_HIVE_training_mean_field_CPU_20261101_v1.py'),
                            ('independent_CPU_audit','audit_HIVE_training_mean_field_CPU_20261101_v1.py')]:
            before=resource_snapshot(s)
            assert before['effective_CPU_headroom_bytes']>=s['minimum_start_CPU_headroom_bytes']
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            command=[sys.executable,'-B',str(M.CODE/entry),'--protocol',str(protocol)]
            start=time.monotonic();largest_drop=0
            with (M.OUT/(phase+'.stdout.log')).open('xb') as stdout,(M.OUT/(phase+'.stderr.log')).open('xb') as stderr:
                owner=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                    cwd=parent['repository_root'],env=env,start_new_session=True)
                ticks=M.process(owner.pid,-1)['start_ticks']
                atomic_progress(dict(PID=os.getpid(),phase=phase,active_PID=owner.pid,active_start_ticks=ticks,
                    protocol=M.ref(protocol),new_GPU_jobs=0,new_training_updates=0))
                while owner.poll() is None:
                    assert time.monotonic()-start<=s['CPU_seconds_per_stage'],'Finite CPU stage timeout'
                    current=resource_snapshot(s)
                    largest_drop=max(largest_drop,before['effective_CPU_headroom_bytes']-current['effective_CPU_headroom_bytes'])
                    assert current['effective_CPU_headroom_bytes']>=s['minimum_runtime_CPU_headroom_bytes']
                    assert largest_drop<=s['maximum_extra_CPU_bytes'],'Shared headroom drop exceeds registered CPU guard'
                    time.sleep(.25)
                rc=owner.wait()
            event=dict(phase=phase,command=command,PID=owner.pid,exit_code=rc,seconds=time.monotonic()-start,
                       resources_before=before,maximum_observed_shared_headroom_drop_bytes=largest_drop,
                       GPU=False,new_training_updates=0)
            owner=None;events.append(event);M.write(M.OUT/(phase+'_call.json'),event)
            assert rc==0
        audit=M.read(M.OUT/'full_CPU_audit.json')
        assert audit['status']=='PASS_BOTH_FULL_TRAIN_RAW_MEANS_AND_CPU_ONLY_INTERFACE_AUDIT'
        M.write(M.OUT/'completion.json',dict(status='COMPLETE_TRAIN_ONLY_MEAN_FIELD_CPU_PREPARATION_NOT_GPU_EFFICACY',
            protocol=M.ref(protocol),events=events,CPU_audit=M.ref(M.OUT/'full_CPU_audit.json'),
            new_training_updates=0,new_GPU_jobs=0,new_reconstruction_images=0,
            R80_inputs_or_image_score_payloads_used=False,GPU_generation_admitted=False,
            second_method_or_SOTA_or_CCF_A_claimed=False,NEW_contacted=False))
    except BaseException as error:
        if owner is not None:
            try:stop_owned(owner)
            except ProcessLookupError:pass
        M.write(M.OUT/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),
            events=events,no_automatic_retry=True,new_GPU_jobs=0))
        raise

if __name__=='__main__':
    def stop(*unused):raise InterruptedError('Stop only this CPU preparation queue and its directly owned child')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);run(p.parse_args().protocol)
