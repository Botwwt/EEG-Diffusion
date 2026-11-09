"""One finite engineering profile after the exact full-latent input stage exits."""
import argparse
import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import brainhive_full1654_ordinary_engine_20261101_v1 as E
K=E.K


def live(pid,ticks):
    p=Path('/proc',str(pid))
    try:fields=(p/'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return False
    return fields[0] not in ('Z','X') and int(fields[19])==ticks


def run(protocol):
    s=E.setup_profile(protocol)
    from brainhive_owned_process_CPU_20261029_v1 import capture_wrapper_child,stop_owned
    root=Path(s['output_root']);assert not root.exists();root.mkdir()
    code=Path(protocol).parent;R=Path(s['repository_root']);up=s['upstream'];dependency=Path(s['latent_output_root'])
    owner=pinned=None;phase='WAIT_EXACT_LATENT_STAGE';events=[]
    K.write(root/'waiting.json',dict(PID=os.getpid(),upstream=up,protocol=K.ref(protocol),GPU_jobs_started=0))
    deadline=datetime.datetime.fromisoformat(s['maximum_wait_deadline_UTC']).timestamp()
    try:
        while live(up['PID'],up['start_ticks']):
            assert time.time()<deadline,'Finite dependency deadline exhausted; do not restart'
            time.sleep(20)
        assert not (dependency/'failure.json').exists()
        done=K.read(dependency/'completion.json')
        assert done['status']=='COMPLETE_EXPOSED_FULL16540_NATIVE_TRAIN_LATENT_INPUT_STAGE' and done['protocol']==s['latent_protocol']
        K.write(root/'dependency_verified.json',dict(completion=K.ref(dependency/'completion.json'),no_method_score_read=True))
        for label,script,gpu in [('GPU_profile','profile_HIVE_full1654_original_ordinary_GPU_20261101_v1.py',True),
                                 ('CPU_audit','audit_HIVE_full1654_ordinary_profile_CPU_20261101_v1.py',False)]:
            phase=label;K.resources(s,starting=gpu)
            command=[sys.executable,'-B',str(code/script),'--protocol',str(protocol)]
            name='HIVE_full1654_original_ordinary_profile_20261101_v1'
            if gpu:
                assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
                assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==K.GPU
                command=[sys.executable,'-B',str(R/'run_gpu_job_v2.py'),'--name',name,'--max-seconds','900','--']+command
            logs=root/'queue';logs.mkdir(exist_ok=True)
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0' if gpu else '',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            start=time.monotonic()
            with (logs/(label+'.stdout.log')).open('xb') as stdout,(logs/(label+'.stderr.log')).open('xb') as stderr:
                owner=subprocess.Popen(command,stdout=stdout,stderr=stderr,stdin=subprocess.DEVNULL,cwd=R,env=env,start_new_session=True)
                pinned=capture_wrapper_child(owner,R/'audit'/(name+'_job.json')) if gpu else None
                K.write(logs/(label+'_started.json'),dict(PID=owner.pid,command=command,GPU=gpu,job_name=name))
                rc=owner.wait(timeout=965 if gpu else 1865)
            event=dict(label=label,command=command,GPU=gpu,exit_code=rc,seconds=time.monotonic()-start)
            if gpu:
                job=K.read(R/'audit'/(name+'_job.json'));assert job['wrapper_pid']==owner.pid and job['exit_code']==rc
                event['GPU_receipt']=K.ref(R/'audit'/(name+'_job.json'))
            owner=pinned=None;events.append(event);K.write(logs/(label+'.json'),event);assert rc==0
        audit=K.read(root/'profile/independent_CPU_audit.json')
        assert audit['status']=='PASS_ACTUAL_FOUR_STEP_ORDINARY_LOSS_HEAD_VJP_AND_ADAM_CPU_AUDIT'
        K.write(root/'completion.json',dict(status='COMPLETE_ORIGINAL_ORDINARY_FULL1654_ENGINEERING_PROFILE_ONLY',
            protocol=K.ref(protocol),events=events,profile=K.ref(root/'profile/completion.json'),CPU_audit=K.ref(root/'profile/independent_CPU_audit.json'),
            temporary_updates=4,formal_training_admitted=False,formal_learned_weights_produced=False,
            new_research_images=0,R80_effects_read=False,independent_H=False,NEW_contacted=False))
    except BaseException as error:
        if owner is not None:stop_owned(owner,pinned)
        K.write(root/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),events=events,no_automatic_retry=True));raise


if __name__=='__main__':
    def stop(*unused):raise InterruptedError('Stop only this owned engineering profile')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);run(parser.parse_args().protocol)
