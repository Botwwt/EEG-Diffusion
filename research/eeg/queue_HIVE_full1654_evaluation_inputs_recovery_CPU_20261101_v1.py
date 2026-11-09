"""Explicit reviewed evaluation-only continuation and zero-GPU dependency recovery.

All old source/protocol/failure bytes stay present. No head fit or research image
generation is repeated. Only the originally allowed evaluator timeout slots run.
"""
import argparse
import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from brainhive_gpu_exit_contract_CPU_20261101_v1 import verify as verify_exit


def run(protocol):
    import json
    r=json.loads(Path(protocol).read_bytes());oldcode=Path(r['original_study']['path']).parent
    sys.path.insert(0,str(oldcode))
    import brainhive_full1654_F_study_contract_20261101_v1 as G
    s=G.setup(r['original_study']['path']);assert G.ref(r['original_study']['path'])==r['original_study']
    for pin in r['source_pins']+[r['original_failure'],r['failed_feature_job'],r['latent_protocol'],r['ordinary_profile_protocol']]:G.check(pin)
    review=G.read(G.check(r['actual_recovery_review']))
    assert review['status']=='PASS_ACTUAL_EVALUATION_ONLY_AND_ZERO_UPDATE_INPUT_CHAIN_RECOVERY_REVIEW'
    assert review['source_pins']==r['source_pins']
    assert r['bounded_feature_attempts']==[1,2] and r['Euclidean_feature_attempts']==[0,1,2]
    root=Path(s['output_root']);out=Path(r['recovery_output_root']);assert not out.exists();out.mkdir()
    logs=out/'queue';logs.mkdir();R=Path(s['repository_root']);owner=pinned=None;events=[];phase='INIT'
    from brainhive_owned_process_CPU_20261029_v1 import capture_wrapper_child,stop_owned
    def resources():
        budget=G.resources(s)
        extra=(2<<30)+(64<<20)+(16<<20)
        assert budget['free_bytes']>=budget['required_remaining_bytes']+extra
        assert G.allocated(out)<=16<<20
    def checkpoint_sources():
        for pin in r['preserved_feature_shards']:G.check(pin)
        for arm,pin in r['final_models'].items():G.check(pin)
        for arm,pin in r['image_manifests'].items():G.check(pin)
    def call(label,args,gpu=False,seconds=1800,allow_timeout=False):
        nonlocal owner,pinned,phase
        phase=label;resources()
        deadline=r['original_evaluation_deadline_UTC'] if label.startswith('eval_') else r['whole_recovery_deadline_UTC']
        assert time.time()<datetime.datetime.fromisoformat(deadline).timestamp()
        command=[sys.executable,'-B']+list(map(str,args));name='HIVE_full1654_recovery_20261101_v1_'+label
        if gpu:
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==G.GPU
            command=[sys.executable,'-B',str(R/'run_gpu_job_v2.py'),'--name',name,'--max-seconds','900','--']+command;seconds=900
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0' if gpu else '',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        tick=time.monotonic()
        with (logs/(label+'.stdout.log')).open('xb') as stdout,(logs/(label+'.stderr.log')).open('xb') as stderr:
            owner=subprocess.Popen(command,stdout=stdout,stderr=stderr,stdin=subprocess.DEVNULL,cwd=R,env=env,start_new_session=True)
            pinned=capture_wrapper_child(owner,R/'audit'/(name+'_job.json')) if gpu else None
            fields=Path('/proc',str(owner.pid),'stat').read_text().rsplit(')',1)[1].split()
            from_json=dict(PID=os.getpid(),phase=phase,active_PID=owner.pid,active_start_ticks=int(fields[19]),
                           GPU=gpu,job_name=name,completed_tasks=len(events),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
            pending=out/'progress.pending.json';pending.write_text(json.dumps(from_json)+'\n');os.replace(pending,out/'progress.json')
            rc=owner.wait(timeout=seconds+65)
        event=dict(label=label,command=command,wrapper_exit_code=rc,GPU=gpu,seconds=time.monotonic()-tick)
        if gpu:
            receipt=G.read(R/'audit'/(name+'_job.json'));event.update(verify_exit(rc,receipt,owner.pid));event['GPU_receipt']=G.ref(R/'audit'/(name+'_job.json'))
        else:event['success']=rc==0
        owner=pinned=None;events.append(event);G.write(logs/(label+'.json'),event)
        if not event['success']:
            assert allow_timeout and event.get('timed_out') is True,'Non-timeout failure; preserve and stop'
            return False
        return True
    def guarded(arm,script,args,label):
        if arm=='bounded_F':execution=oldcode/'bounded_F_evaluation_execution.json'
        else:execution=Path(protocol).parent/'Euclidean_F_evaluation_execution.json'
        return [oldcode/'run_HIVE_R80_evaluation_guarded_20261029_v1.py','--execution',execution,
                '--audit-file',root/arm/'evaluation_resources'/(label+'.json'),'--script',script,'--']+list(args)
    def preserve_partial(folder,label):
        for path in folder.rglob('*.tmp'):
            assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(folder.resolve())
            destination=path.with_name(path.name+'.'+label+'_incomplete_retained');assert not destination.exists();path.rename(destination)
    try:
        checkpoint_sources()
        for arm,attempts in [('bounded_F',r['bounded_feature_attempts']),('Euclidean_F',r['Euclidean_feature_attempts'])]:
            feature=root/arm/'evaluation/features';manifest=root/arm/'loss_control/manifest.jsonl'
            for part in attempts:
                label=f'eval_features_{arm}_{part}'
                if call(label,guarded(arm,R/'extract_image_features.py',['--manifest',manifest,'--output',feature],label),True,allow_timeout=True):break
                preserve_partial(feature,label);checkpoint_sources()
            else:raise RuntimeError('Original fixed3 feature slots exhausted; no additional slot')
            assert G.read(feature/'completion.json')['queries']==2000;checkpoint_sources()
            for label,script,args in [('eval_common_'+arm,R/'score_full_manifest.py',['--manifest',manifest,'--features',feature,
                   '--gallery-manifest',s['evaluation']['gallery_manifest']['path'],'--gallery-features',s['evaluation']['gallery_features'],'--output',root/arm/'evaluation/scores']),
                   ('eval_low_'+arm,R/'score_low_level.py',['--manifest',manifest,'--gallery-manifest',s['evaluation']['gallery_manifest']['path'],'--output',root/arm/'evaluation/low_level'])]:
                call(label,guarded(arm,script,args,label))
            for part in range(3):
                label=f'eval_author_{arm}_{part}'
                execution=oldcode/'bounded_F_evaluation_execution.json' if arm=='bounded_F' else Path(protocol).parent/'Euclidean_F_evaluation_execution.json'
                if call(label,guarded(arm,oldcode/'score_HIVE_loss_control_R80_author_GPU_20261030_v1.py',['--execution',execution],label),True,allow_timeout=True):break
            else:raise RuntimeError('Original fixed3 author slots exhausted')
        call('eval_compare_all16',[oldcode/'compare_HIVE_full1654_F_R80_CPU_20261101_v1.py','--study',r['original_study']['path']])
        assert time.time()<datetime.datetime.fromisoformat(r['original_evaluation_deadline_UTC']).timestamp()
        checkpoint_sources()
        phase_completion=dict(status='COMPLETE_ORIGINAL_STUDY_EVALUATION_BY_REVIEWED_TIMEOUT_RECOVERY',
            recovery_protocol=G.ref(protocol),original_study=r['original_study'],original_failure=r['original_failure'],
            original_failed_feature_job=r['failed_feature_job'],joint_summary=G.ref(root/'joint_comparison/summary.json'),
            paired_differences=G.ref(root/'joint_comparison/all3x16_paired_differences.csv.gz'),events=events.copy(),
            no_models_retrained=True,no_research_images_regenerated=True,all_original_feature_prefix_shards_unchanged=True,
            original_evaluation_deadline_unchanged=True)
        G.write(out/'evaluation_completion.json',phase_completion)
        G.write(root/'completion.json',dict(status='COMPLETE_FULL1654_TWO_F_FITS_4000_R80_IMAGES_ALL16_FIXED_COMPARISONS',
            study=r['original_study'],events=G.read(r['original_failure']['path'])['events']+events.copy(),
            original_controller_failed=True,original_failure_preserved=r['original_failure'],
            actual_evaluation_recovery=G.ref(out/'evaluation_completion.json'),actual_recovery_controller_PID=os.getpid(),
            training_seed=1729,new_head_updates=62028,new_EEG_exposures=992400,new_research_images=4000,
            reference_native_images_reused=2000,official200_previously_exposed=True,independent_H=False,
            NEW_contacted=False,SOTA=False,CCF_A=False))
        # Old waiters did zero GPU work. Reuse their unchanged producers/protocols,
        # while this new reviewed controller records the explicit dependency fix.
        latent=G.read(r['latent_protocol']['path']);lcode=Path(r['latent_protocol']['path']).parent;lroot=Path(latent['output_root'])
        for label,args in [('latent_profile',['--mode','profile']),('latent_segment0',['--mode','build','--segment','0']),
                           ('latent_segment1',['--mode','build','--segment','1'])]:
            call(label,[lcode/'build_HIVE_exposed_full16540_latents_GPU_20261101_v1.py','--protocol',r['latent_protocol']['path']]+args,True)
        call('latent_CPU_audit',[lcode/'audit_HIVE_exposed_full16540_latents_CPU_20261101_v1.py','--protocol',r['latent_protocol']['path']])
        audit=G.read(lroot/'independent_CPU_audit.json');assert audit['status']=='PASS_ALL16540_LATENT_CHUNKS_AND_ALL13230_OLD_T_VALUE_REPLAY'
        G.write(lroot/'completion.json',dict(status='COMPLETE_EXPOSED_FULL16540_NATIVE_TRAIN_LATENT_INPUT_STAGE',
            protocol=r['latent_protocol'],bank=audit['bank'],CPU_audit=G.ref(lroot/'independent_CPU_audit.json'),
            original_waiter_failure_preserved=G.ref(lroot/'failure.json'),actual_recovery_protocol=G.ref(protocol),
            actual_recovery_controller_PID=os.getpid(),events=[v for v in events if v['label'].startswith('latent_')],
            parameter_updates=0,new_research_images=0,official_test_image_payloads_opened=0,
            official200_previously_exposed=True,independent_H=False,NEW_contacted=False))
        o=G.read(r['ordinary_profile_protocol']['path']);ocode=Path(r['ordinary_profile_protocol']['path']).parent;oroot=Path(o['output_root'])
        call('ordinary_GPU_profile',[ocode/'profile_HIVE_full1654_original_ordinary_GPU_20261101_v1.py','--protocol',r['ordinary_profile_protocol']['path']],True)
        call('ordinary_CPU_profile_audit',[ocode/'audit_HIVE_full1654_ordinary_profile_CPU_20261101_v1.py','--protocol',r['ordinary_profile_protocol']['path']])
        assert G.read(oroot/'profile/independent_CPU_audit.json')['status']=='PASS_ACTUAL_FOUR_STEP_ORDINARY_LOSS_HEAD_VJP_AND_ADAM_CPU_AUDIT'
        G.write(oroot/'completion.json',dict(status='COMPLETE_ORIGINAL_ORDINARY_FULL1654_ENGINEERING_PROFILE_ONLY',
            protocol=r['ordinary_profile_protocol'],profile=G.ref(oroot/'profile/completion.json'),CPU_audit=G.ref(oroot/'profile/independent_CPU_audit.json'),
            original_waiter_failure_preserved=G.ref(oroot/'failure.json'),actual_recovery_protocol=G.ref(protocol),
            actual_recovery_controller_PID=os.getpid(),temporary_updates=4,formal_training_admitted=False,
            formal_learned_weights_produced=False,new_research_images=0,R80_effects_read=False,independent_H=False,NEW_contacted=False))
        G.write(out/'completion.json',dict(status='COMPLETE_REVIEWED_EVALUATION_AND_INPUT_ENGINEERING_RECOVERY_CHAIN',
            protocol=G.ref(protocol),events=events,F_evaluation=G.ref(root/'completion.json'),
            latent_input=G.ref(lroot/'completion.json'),ordinary_engineering=G.ref(oroot/'completion.json'),
            no_F_models_retrained=True,no_research_images_regenerated=True,formal_ordinary_fit_admitted=False,NEW_contacted=False))
    except BaseException as error:
        if owner is not None:stop_owned(owner,pinned)
        G.write(out/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),events=events,no_automatic_retry=True));raise


if __name__=='__main__':
    def stop(*unused):raise InterruptedError('Stop only the owned recovery chain')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);run(parser.parse_args().protocol)
