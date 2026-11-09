"""Only previously unexecuted native-input/profile work after verified cache release."""
import argparse
import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from brainhive_gpu_exit_contract_CPU_20261101_v1 import verify

def run(protocol):
    import json
    q=json.loads(Path(protocol).read_bytes());old=Path(q['original_F_study']['path']).parent
    sys.path.insert(0,str(old));import brainhive_full1654_F_study_contract_20261101_v1 as G
    s=G.setup(q['original_F_study']['path'])
    assert q['status']=='REVIEWED_INPUT_ONLY_RECOVERY_AFTER_ZERO_CUDA_HEADROOM_FAILURE'
    for pin in q['source_pins']+[q['original_input_failed_job'],q['actual_cache_advice'],q['completed_F_study'],q['actual_review']]:G.check(pin)
    R=Path(s['repository_root']);root=Path(q['output_root']);assert not root.exists();root.mkdir();logs=root/'queue';logs.mkdir()
    from brainhive_owned_process_CPU_20261029_v1 import capture_wrapper_child,stop_owned
    events=[];owner=pinned=None;phase='INIT'
    def call(label,args,gpu=False):
        nonlocal owner,pinned,phase
        phase=label;assert time.time()<datetime.datetime.fromisoformat(q['original_deadline_UTC']).timestamp()
        G.resources(s);assert G.allocated(root)<16<<20
        command=[sys.executable,'-B']+list(map(str,args));name='HIVE_input_memory_recovery_20261101_v1_'+label
        if gpu:
            assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==G.GPU
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            command=[sys.executable,'-B',str(R/'run_gpu_job_v2.py'),'--name',name,'--max-seconds','900','--']+command
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0' if gpu else '',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        if gpu and label.startswith('latent_'):
            # The unchanged VAE producer explicitly requests deterministic Torch
            # algorithms. Supply their deterministic CUDA BLAS workspace.
            env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        started=time.monotonic()
        with (logs/(label+'.stdout.log')).open('xb') as stdout,(logs/(label+'.stderr.log')).open('xb') as stderr:
            owner=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,cwd=R,env=env,start_new_session=True)
            pinned=capture_wrapper_child(owner,R/'audit'/(name+'_job.json')) if gpu else None
            f=Path('/proc',str(owner.pid),'stat').read_text().rsplit(')',1)[1].split()
            pending=root/'progress.pending.json';pending.write_text(json.dumps(dict(PID=os.getpid(),phase=label,active_PID=owner.pid,
                active_start_ticks=int(f[19]),GPU=gpu,job_name=name,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))+'\n');os.replace(pending,root/'progress.json')
            rc=owner.wait(timeout=965 if gpu else 1800)
        event=dict(label=label,command=command,GPU=gpu,wrapper_exit_code=rc,seconds=time.monotonic()-started)
        if gpu:
            pin=G.ref(R/'audit'/(name+'_job.json'));event.update(verify(rc,G.read(pin['path']),owner.pid));event['GPU_receipt']=pin
        else:event['success']=rc==0
        owner=pinned=None;events.append(event);G.write(logs/(label+'.json'),event)
        assert event['success'],'Preserve failure; no automatic retry'
    try:
        lpin=q['latent_protocol'];lcode=Path(lpin['path']).parent;l=G.read(G.check(lpin));lroot=Path(l['output_root'])
        for label,args in [('latent_profile',['--mode','profile']),('latent_segment0',['--mode','build','--segment','0']),('latent_segment1',['--mode','build','--segment','1'])]:
            call(label,[lcode/'build_HIVE_exposed_full16540_latents_GPU_20261101_v1.py','--protocol',lpin['path']]+args,True)
        call('latent_CPU_audit',[lcode/'audit_HIVE_exposed_full16540_latents_CPU_20261101_v1.py','--protocol',lpin['path']])
        a=G.read(lroot/'independent_CPU_audit.json');assert a['status']=='PASS_ALL16540_LATENT_CHUNKS_AND_ALL13230_OLD_T_VALUE_REPLAY'
        G.write(lroot/'completion.json',dict(status='COMPLETE_EXPOSED_FULL16540_NATIVE_TRAIN_LATENT_INPUT_STAGE',protocol=lpin,
            bank=a['bank'],CPU_audit=G.ref(lroot/'independent_CPU_audit.json'),original_waiter_failure_preserved=G.ref(lroot/'failure.json'),
            additional_zero_CUDA_headroom_failure_preserved=q['original_input_failed_job'],actual_recovery_protocol=G.ref(protocol),
            actual_recovery_controller_PID=os.getpid(),parameter_updates=0,new_research_images=0,official_test_image_payloads_opened=0,
            official200_previously_exposed=True,independent_H=False,NEW_contacted=False))
        opin=q['ordinary_profile_protocol'];ocode=Path(opin['path']).parent;o=G.read(G.check(opin));oroot=Path(o['output_root'])
        call('ordinary_GPU_profile',[ocode/'profile_HIVE_full1654_original_ordinary_GPU_20261101_v1.py','--protocol',opin['path']],True)
        call('ordinary_CPU_audit',[ocode/'audit_HIVE_full1654_ordinary_profile_CPU_20261101_v1.py','--protocol',opin['path']])
        assert G.read(oroot/'profile/independent_CPU_audit.json')['status']=='PASS_ACTUAL_FOUR_STEP_ORDINARY_LOSS_HEAD_VJP_AND_ADAM_CPU_AUDIT'
        G.write(oroot/'completion.json',dict(status='COMPLETE_ORIGINAL_ORDINARY_FULL1654_ENGINEERING_PROFILE_ONLY',protocol=opin,
            profile=G.ref(oroot/'profile/completion.json'),CPU_audit=G.ref(oroot/'profile/independent_CPU_audit.json'),
            original_waiter_failure_preserved=G.ref(oroot/'failure.json'),actual_recovery_protocol=G.ref(protocol),actual_recovery_controller_PID=os.getpid(),
            temporary_updates=4,formal_training_admitted=False,formal_learned_weights_produced=False,new_research_images=0,R80_effects_read=False,independent_H=False,NEW_contacted=False))
        G.write(root/'completion.json',dict(status='COMPLETE_REVIEWED_EVALUATION_AND_INPUT_ENGINEERING_RECOVERY_CHAIN',protocol=G.ref(protocol),events=events,
            F_evaluation=q['completed_F_study'],original_evaluation_recovery_protocol=q['original_evaluation_recovery_protocol'],
            latent_input=G.ref(lroot/'completion.json'),ordinary_engineering=G.ref(oroot/'completion.json'),F_evaluation_reused_without_rerun=True,
            no_F_models_retrained=True,no_research_images_regenerated=True,formal_ordinary_fit_admitted=False,NEW_contacted=False))
    except BaseException as error:
        if owner is not None:stop_owned(owner,pinned)
        G.write(root/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),events=events,no_automatic_retry=True));raise

if __name__=='__main__':
    def stop(*unused):raise InterruptedError('Stop only owned native-input recovery')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);run(parser.parse_args().protocol)
