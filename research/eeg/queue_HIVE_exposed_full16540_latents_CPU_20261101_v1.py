"""Wait the same existing complete lifecycle, then one VAE profile/two fixed segments."""
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
import brainhive_exposed_native_latent_contract_CPU_20261101_v1 as K


def exact_process(pid,ticks):
    path=Path('/proc',str(pid))
    if not path.exists():return False
    try:
        fields=(path/'stat').read_text().rsplit(')',1)[1].split()
        return fields[0] not in ('Z','X') and int(fields[19])==ticks
    except FileNotFoundError:return False


def run(protocol):
    s=K.setup(protocol)
    from brainhive_owned_process_CPU_20261029_v1 import capture_wrapper_child,stop_owned
    root=Path(s['output_root']);assert not root.exists();root.mkdir()
    code=Path(protocol).parent;R=Path(s['repository_root']);up=s['upstream'];K.check(up['study'])
    dependency=Path(up['output_root']);events=[];owner=pinned=None;phase='WAIT_EXACT_UPSTREAM'
    K.write(root/'waiting.json',dict(PID=os.getpid(),dependency=up,maximum_wait_deadline_UTC=s['maximum_wait_deadline_UTC'],
            GPU_jobs_started=0,source=K.ref(protocol),effects_not_required_for_latent_stage=True))
    deadline=datetime.datetime.fromisoformat(s['maximum_wait_deadline_UTC']).timestamp()
    try:
        while exact_process(up['PID'],up['start_ticks']):
            assert time.time()<deadline,'Finite original upstream deadline exhausted; do not restart upstream'
            time.sleep(20)
        assert not (dependency/'failure.json').exists(),'Upstream failed; retain and stop without GPU'
        complete=K.read(dependency/'completion.json')
        assert complete['status']=='COMPLETE_FULL1654_TWO_F_FITS_4000_R80_IMAGES_ALL16_FIXED_COMPARISONS'
        assert complete['new_research_images']==4000 and complete['study']==up['study']
        K.write(root/'dependency_verified.json',dict(exact_original_controller_terminal=True,
            upstream_completion=K.ref(dependency/'completion.json'),no_method_effect_payload_read=True))
        K.resources(s,starting=True)
        started=time.monotonic()
        def call(label,args,gpu):
            nonlocal owner,pinned,phase
            phase=label;assert time.monotonic()-started<5400
            K.resources(s,starting=gpu)
            name='HIVE_exposed16540_latents_20261101_v1_'+label
            command=[sys.executable,'-B']+list(map(str,args))
            seconds=1800
            if gpu:
                assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
                assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==K.GPU
                command=[sys.executable,'-B',str(R/'run_gpu_job_v2.py'),'--name',name,'--max-seconds','900','--']+command
                seconds=900
            logs=root/'queue';logs.mkdir(exist_ok=True)
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0' if gpu else '',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            tick=time.monotonic()
            with (logs/(label+'.stdout.log')).open('xb') as out,(logs/(label+'.stderr.log')).open('xb') as err:
                owner=subprocess.Popen(command,stdout=out,stderr=err,stdin=subprocess.DEVNULL,cwd=R,env=env,start_new_session=True)
                pinned=capture_wrapper_child(owner,R/'audit'/(name+'_job.json')) if gpu else None
                K.write(logs/(label+'_started.json'),dict(PID=owner.pid,command=command,GPU=gpu,job_name=name))
                rc=owner.wait(timeout=seconds+65)
            event=dict(label=label,command=command,exit_code=rc,seconds=time.monotonic()-tick,GPU=gpu)
            if gpu:
                job=K.read(R/'audit'/(name+'_job.json'))
                assert job['wrapper_pid']==owner.pid and job['exit_code']==rc
                event['GPU_receipt']=K.ref(R/'audit'/(name+'_job.json'))
            owner=pinned=None;events.append(event);K.write(logs/(label+'.json'),event)
            assert rc==0,'Finite input stage failed; no automatic retry'
        call('profile',[code/'build_HIVE_exposed_full16540_latents_GPU_20261101_v1.py','--protocol',protocol,'--mode','profile'],True)
        for segment in (0,1):
            call('segment'+str(segment),[code/'build_HIVE_exposed_full16540_latents_GPU_20261101_v1.py','--protocol',protocol,'--mode','build','--segment',segment],True)
        call('CPU_audit',[code/'audit_HIVE_exposed_full16540_latents_CPU_20261101_v1.py','--protocol',protocol],False)
        audit=K.read(root/'independent_CPU_audit.json')
        assert audit['status']=='PASS_ALL16540_LATENT_CHUNKS_AND_ALL13230_OLD_T_VALUE_REPLAY'
        K.write(root/'completion.json',dict(status='COMPLETE_EXPOSED_FULL16540_NATIVE_TRAIN_LATENT_INPUT_STAGE',
                protocol=K.ref(protocol),events=events,bank=audit['bank'],CPU_audit=K.ref(root/'independent_CPU_audit.json'),
                component_training_admitted=False,parameter_updates=0,new_research_images=0,
                official_test_image_payloads_opened=0,seconds_after_upstream=time.monotonic()-started,
                official200_previously_exposed=True,independent_H=False,NEW_contacted=False,SOTA=False))
    except BaseException as error:
        if owner is not None:stop_owned(owner,pinned)
        K.write(root/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),events=events,no_automatic_retry=True))
        raise


if __name__=='__main__':
    def stop(*unused):raise InterruptedError('Stop this owned latent input stage only')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True)
    run(parser.parse_args().protocol)
