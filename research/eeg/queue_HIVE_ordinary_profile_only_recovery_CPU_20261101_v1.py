"""Reuse completed inputs; run only the still-unexecuted four-update engineering."""
import argparse,datetime,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from brainhive_gpu_exit_contract_CPU_20261101_v1 import verify

def run(protocol):
    import json
    q=json.loads(Path(protocol).read_bytes());sys.path.insert(0,str(Path(q['original_F_study']['path']).parent))
    import brainhive_full1654_F_study_contract_20261101_v1 as G
    s=G.setup(q['original_F_study']['path'])
    assert q['status']=='REVIEWED_ORDINARY_PROFILE_ONLY_AFTER_COMPLETE_NATIVE_INPUTS'
    for pin in q['source_pins']+[q['actual_review'],q['actual_cache_advice'],q['completed_F_study'],q['native_input_completion'],q['preserved_previous_recovery_failure']]:G.check(pin)
    from brainhive_owned_process_CPU_20261029_v1 import capture_wrapper_child,stop_owned
    R=Path(s['repository_root']);root=Path(q['output_root']);assert not root.exists();root.mkdir();logs=root/'queue';logs.mkdir()
    events=[];owner=pinned=None;phase='INIT'
    def call(label,args,gpu=False):
        nonlocal owner,pinned,phase
        phase=label;assert time.time()<datetime.datetime.fromisoformat(q['original_deadline_UTC']).timestamp()
        G.resources(s);assert G.allocated(root)<16<<20
        command=[sys.executable,'-B']+list(map(str,args));name='HIVE_ordinary_profile_only_recovery_20261101_v1_'+label
        if gpu:
            assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==G.GPU
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            command=[sys.executable,'-B',str(R/'run_gpu_job_v2.py'),'--name',name,'--max-seconds','900','--']+command
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0' if gpu else '',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        started=time.monotonic()
        with (logs/(label+'.stdout.log')).open('xb') as stdout,(logs/(label+'.stderr.log')).open('xb') as stderr:
            owner=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,cwd=R,env=env,start_new_session=True)
            pinned=capture_wrapper_child(owner,R/'audit'/(name+'_job.json')) if gpu else None
            fields=Path('/proc',str(owner.pid),'stat').read_text().rsplit(')',1)[1].split()
            pending=root/'progress.pending.json';pending.write_text(json.dumps(dict(PID=os.getpid(),phase=label,active_PID=owner.pid,
                active_start_ticks=int(fields[19]),GPU=gpu,job_name=name,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))+'\n');os.replace(pending,root/'progress.json')
            rc=owner.wait(timeout=965 if gpu else 1800)
        event=dict(label=label,command=command,GPU=gpu,wrapper_exit_code=rc,seconds=time.monotonic()-started)
        if gpu:
            pin=G.ref(R/'audit'/(name+'_job.json'));event.update(verify(rc,G.read(pin['path']),owner.pid));event['GPU_receipt']=pin
        else:event['success']=rc==0
        owner=pinned=None;events.append(event);G.write(logs/(label+'.json'),event)
        assert event['success'],'Preserve failure; no automatic retry'
    try:
        opin=q['ordinary_profile_protocol'];ocode=Path(opin['path']).parent;o=G.read(G.check(opin));oroot=Path(o['output_root'])
        assert not (oroot/'profile').exists()
        native=G.read(G.check(q['native_input_completion']));assert native['protocol']==q['latent_protocol']
        assert native['status']=='COMPLETE_EXPOSED_FULL16540_NATIVE_TRAIN_LATENT_INPUT_STAGE'
        call('ordinary_GPU_profile',[ocode/'profile_HIVE_full1654_original_ordinary_GPU_20261101_v1.py','--protocol',opin['path']],True)
        call('ordinary_CPU_audit',[ocode/'audit_HIVE_full1654_ordinary_profile_CPU_20261101_v1.py','--protocol',opin['path']])
        assert G.read(oroot/'profile/independent_CPU_audit.json')['status']=='PASS_ACTUAL_FOUR_STEP_ORDINARY_LOSS_HEAD_VJP_AND_ADAM_CPU_AUDIT'
        G.write(oroot/'completion.json',dict(status='COMPLETE_ORIGINAL_ORDINARY_FULL1654_ENGINEERING_PROFILE_ONLY',protocol=opin,
            profile=G.ref(oroot/'profile/completion.json'),CPU_audit=G.ref(oroot/'profile/independent_CPU_audit.json'),
            original_waiter_failure_preserved=G.ref(oroot/'failure.json'),additional_zero_CUDA_headroom_failure_preserved=q['preserved_previous_recovery_failure'],
            actual_recovery_protocol=G.ref(protocol),actual_recovery_controller_PID=os.getpid(),temporary_updates=4,
            formal_training_admitted=False,formal_learned_weights_produced=False,new_research_images=0,R80_effects_read=False,independent_H=False,NEW_contacted=False))
        G.write(root/'completion.json',dict(status='COMPLETE_REVIEWED_EVALUATION_AND_INPUT_ENGINEERING_RECOVERY_CHAIN',protocol=G.ref(protocol),events=events,
            F_evaluation=q['completed_F_study'],original_evaluation_recovery_protocol=q['original_evaluation_recovery_protocol'],
            latent_input=q['native_input_completion'],ordinary_engineering=G.ref(oroot/'completion.json'),F_evaluation_reused_without_rerun=True,
            completed_native_input_bank_reused_without_rebuild=True,no_F_models_retrained=True,no_research_images_regenerated=True,
            formal_ordinary_fit_admitted=False,NEW_contacted=False))
    except BaseException as error:
        if owner is not None:stop_owned(owner,pinned)
        G.write(root/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),events=events,no_automatic_retry=True));raise

if __name__=='__main__':
    def stop(*unused):raise InterruptedError('Stop only owned profile-only recovery')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);run(parser.parse_args().protocol)
