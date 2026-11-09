"""One finite public-runtime transfer -> CPU install -> GPU replay sequence."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
PERSISTENT=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')
START=time.monotonic()


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_bytes())


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())


def state(pid):
    folder=Path('/proc')/str(pid)
    if not folder.exists():return None
    value=(folder/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,state=value[0],start_ticks=int(value[19]),argv=(folder/'cmdline').read_bytes().decode().replace('\0',' '))


def run(spec_path):
    spec=read(spec_path);assert spec['schema']=='ADDITIONAL_HOST_NATIVE_HIVE_ENGINEERING_V1'
    output=Path(spec['queue_output']);assert output.is_relative_to(PERSISTENT/'native_HIVE_runtime_20261028_v1')
    assert not output.exists();output.mkdir(parents=True)
    complete=[];current='WAIT_EXISTING_TRANSFER'
    with (output/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            def verify(require_all=False):
                assert time.monotonic()-START<9000
                for name,digest in spec['source_sha256'].items():
                    path=ROOT/name
                    if path.exists() or require_all:assert sha(path)==digest,name
            verify()
            launch=read(spec['transfer_launch']);expected=spec['expected_transfer']
            assert launch['pid']==expected['pid'] and launch['process_identity']['start_ticks']==expected['start_ticks']
            assert launch['source_sha256']==expected['source_sha256']
            assert launch['manifest_sha256']==spec['transfer_plan_sha256']==sha(spec['transfer_plan'])
            receipt=Path(spec['transfer_completion'])
            while True:
                verify();observed=state(expected['pid'])
                if observed is None or observed['state'] in ['Z','X','x']:break
                assert observed['start_ticks']==expected['start_ticks'] and observed['argv']==launch['process_identity']['argv']
                assert time.monotonic()-START<7500,'Finite transfer wait expired; no retry'
                time.sleep(2)
            done=read(receipt)
            assert done['status']=='NATIVE_HIVE_PUBLIC_RUNTIME_ASSETS_DIRECT_TRANSFER_ALL_SHA_VERIFIED'
            assert done['manifest_sha256']==spec['transfer_plan_sha256']
            complete.append(dict(stage=current,receipt_sha256=sha(receipt)))
            stages=[('CPU_INSTALL',660,[sys.executable,'-B',str(ROOT/'install_HIVE_second_host_runtime_CPU_20261028_v1.py'),
                         '--plan',spec['transfer_plan'],'--receipt',str(receipt),'--output',str(Path(spec['installation_completion']).parent)]),
                    ('GPU_ENGINEERING',960,[sys.executable,'-B',str(ROOT/'run_BrainHIVE_two_host_GPU_job_20261028_v1.py'),
                         '--name',spec['GPU_job_name'],'--max-seconds','900','--',sys.executable,'-B',
                         str(ROOT/'verify_HIVE_second_host_native_GPU_20261028_v1.py'),'--spec',str(spec_path)])]
            for name,maximum,command in stages:
                verify(require_all=name=='GPU_ENGINEERING');current=name
                argv=['/usr/bin/timeout','--signal=TERM','--kill-after=20',str(maximum),*command]
                before=time.monotonic()
                with (output/(name+'.stdout.log')).open('xb') as out,(output/(name+'.stderr.log')).open('xb') as err:
                    child=subprocess.Popen(argv,cwd=ROOT,stdout=out,stderr=err,start_new_session=True)
                    write(output/(name+'_actual_launch.json'),dict(command=argv,pid=child.pid,process_identity=state(child.pid),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),maximum_seconds=maximum))
                    try:code=child.wait(timeout=maximum+45)
                    except BaseException:
                        if child.poll() is None:
                            os.killpg(child.pid,signal.SIGTERM)
                            try:child.wait(timeout=25)
                            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
                        raise
                record=dict(stage=name,exit_code=code,seconds=time.monotonic()-before)
                write(output/(name+'_actual_terminal.json'),record);assert code==0,record
                path=Path(spec['installation_completion']) if name=='CPU_INSTALL' else Path(spec['output'])/'completion.json'
                result=read(path)
                expected_status='ADDITIONAL_HOST_NATIVE_HIVE_RUNTIME_INSTALLED_AND_STRICT_CPU_PRIORS_VERIFIED' if name=='CPU_INSTALL' else 'ADDITIONAL_HOST_HIVE_NATIVE_PIXEL_AND_FROZEN_ORDINARY_ENGINEERING_PASSED'
                assert result['status']==expected_status
                complete.append(record|dict(completion_sha256=sha(path)))
            verify(require_all=True)
            write(output/'completion.json',dict(status='SECOND_HOST_NATIVE_HIVE_RUNTIME_AND_FOUR_IMAGE_ENGINEERING_COMPLETE',
                  UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),spec_sha256=sha(spec_path),stages=complete,
                  seconds=time.monotonic()-START,new_training_updates=0,new_research_images=0,no_automatic_new_scientific_fit=True))
        except BaseException as error:
            write(output/('failure_'+str(time.time_ns())+'.json'),dict(status='SECOND_HOST_RUNTIME_SEQUENCE_FAILED_PRESERVED_NO_AUTORETRY',stage=current,
                  error=repr(error),traceback=traceback.format_exc(),completed_stages=complete,seconds=time.monotonic()-START))
            raise


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('Explicit runtime sequence termination')))
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);a=p.parse_args();run(a.spec)
