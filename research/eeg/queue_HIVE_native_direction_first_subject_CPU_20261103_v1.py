"""Versioned stagewise scheduling of the unchanged native EEG training recipe."""
import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path


def process(owner, command_fragment=None):
    p=Path('/proc',str(owner['PID']))
    try:
        st=(p/'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:
        return None
    assert int(st[19])==owner['start_ticks'], 'PID reused; do not signal it'
    if command_fragment and st[0] not in ('Z','X'):
        for _ in range(20):
            try:command=(p/'cmdline').read_bytes()
            except FileNotFoundError:return None
            if command:
                assert command_fragment.encode() in command;break
            time.sleep(.01)
            try:st=(p/'stat').read_text().rsplit(')',1)[1].split()
            except FileNotFoundError:return None
            assert int(st[19])==owner['start_ticks']
            if st[0] in ('Z','X'):return st[0]
        else:raise AssertionError('Persistently empty live cmdline')
    return st[0]


def next_segment(root):
    directories=sorted(Path(root).glob('segment_*'))
    numbers=[int(p.name.rsplit('_',1)[1]) for p in directories]
    assert numbers==list(range(len(numbers))), numbers
    for p in directories:
        assert (p/'completion.json').is_file() and not (p/'failure.json').exists(), str(p)
    return len(numbers)


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);a=p.parse_args()
    q=json.loads(Path(a.protocol).read_bytes())
    assert q['status']=='REGISTERED_STAGEWISE_NATIVE_EEG_FIRST_SUBJECT_THEN_MATCHED_IMAGE_REVIEW'
    sys.path.insert(0,str(Path(q['training_protocol']['path']).parent))
    import brainhive_native_direction_full_contract_20261103_v1 as C
    C.checked(q['training_protocol']);s,e,old,base,G,H=C.setup(q['training_protocol']['path'])
    for pin in q['source_pins']:C.checked(pin)
    assert q['host']=='old' and q['seed']==1729 and q['whole_deadline_epoch']>time.time()
    assert q['first_subject']=='sub-01' and q['no_training_recipe_change'] and q['remaining_subjects_only_after_fixed_matched_image_signal']
    root=Path(q['output_root']);out=root/'queue';out.mkdir();worker=None;jobpath=None;active=None
    def stop(sig,frame):raise SystemExit('Stagewise coordinator signal '+str(sig))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def log(stage,**other):
        record=dict(stage=stage,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),**other)
        C.atomic_json(out/'progress.json',record)
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(record)+'\n');f.flush()
        print(json.dumps(record),flush=True)
    def verify_model(subject,arm):
        path=Path(s['output_root'])/'models'/subject/arm/'completion.json'
        rec=C.read(path)
        assert rec['status']=='COMPLETE_FIXED_EPOCH3_NATIVE_EEG_MODEL' and rec['protocol']==q['training_protocol']
        assert rec['subject']==subject and rec['arm']==arm and rec['updates']==2481 and rec['primary_exposures']==39690
        assert rec['epoch']==3 and rec['coverage_count']==39690 and rec['ordinary_frozen'] and rec['host_frozen'] and rec['output_LN_affine_frozen']
        C.checked(rec['final_model']);return C.ref(path),rec['final_model']
    def fit(subject,arm):
        nonlocal worker,jobpath,active
        modelroot=Path(s['output_root'])/'models'/subject/arm
        if (modelroot/'completion.json').exists():return verify_model(subject,arm)
        for segment in range(next_segment(modelroot),s['maximum_segments_per_model']):
            assert time.time()<q['whole_deadline_epoch']-1000
            C.resources(s,e,True)
            name=f'HIVE_native_direction_priority_20261103_{subject}_{arm}_seg{segment:02d}'
            active=f'{subject}_{arm}_segment{segment:03d}'
            jobpath=Path(base['repository_root'])/'audit'/(name+'_job.json')
            cmd=[sys.executable,'-B',str(Path(base['repository_root'])/'run_gpu_job_v2.py'),'--name',name,'--max-seconds','900','--',sys.executable,'-B',str(Path(q['training_protocol']['path']).parent/'train_HIVE_native_direction_full_T_GPU_20261103_v1.py'),'--protocol',q['training_protocol']['path'],'--subject',subject,'--arm',arm,'--segment',str(segment),'--job-name',name]
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
            with (out/(active+'.stdout.log')).open('xb') as stdout,(out/(active+'.stderr.log')).open('xb') as stderr:
                worker=subprocess.Popen(cmd,cwd=Path(a.protocol).parent,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
                C.atomic_json(out/'current.json',dict(PID=worker.pid,command=cmd,name=name,subject=subject,arm=arm,segment=segment))
                log('TRAINING',subject=subject,arm=arm,segment=segment,wrapper_PID=worker.pid)
                rc=worker.wait(timeout=990)
            job=C.read(jobpath);assert rc==0 and job['status']=='COMPLETED' and job['exit_code']==0 and job['wrapper_pid']==worker.pid
            donepath=modelroot/f'segment_{segment:03d}'/'completion.json';done=C.read(donepath)
            assert done['subject']==subject and done['arm']==arm and done['updates_this_segment']>0
            G.write(out/(active+'_closed.json'),dict(segment=C.ref(donepath),job=C.ref(jobpath),wrapper_returncode=rc))
            worker=None
            if done['status']=='COMPLETE_FIXED_EPOCH3_NATIVE_EEG_MODEL':return verify_model(subject,arm)
            assert done['status']=='RESUMABLE_FIXED_NATIVE_EEG_SEGMENT_SOFT_LIMIT'
        raise RuntimeError('Original finite per-model segment limit reached; no reset')
    try:
        launchpath=Path(q['code_root'])/'actual_launch.json'
        for _ in range(50):
            if launchpath.exists():break
            time.sleep(.1)
        launch=C.read(launchpath);assert launch['protocol']==C.ref(a.protocol)
        assert launch['owner']['PID']==os.getpid() and process(launch['owner']) not in (None,'Z','X')
        handoff=C.read(C.checked(launch['handoff']))
        assert handoff['status']=='ORIGINAL_CONTROLLER_RETIRED_AFTER_OWN_WORKER_NORMAL_SOFT_OR_FINAL_BOUNDARY'
        assert process(q['original_training_owner']) in (None,'Z','X')
        assert process(q['original_postfit_owner']) in (None,'Z','X')
        C.checked(q['image_protocol_pin']) if 'image_protocol_pin' in q else C.checked(launch['image_protocol'])
        assert launch['image_protocol']['path']==q['image_protocol_path']
        C.resources(s,e,True)
        models={};finals=[]
        for arm in C.ARMS:
            pin,weight=fit('sub-01',arm);finals.append(pin);models['sub-01/'+arm]=weight
        ready=dict(status='COMPLETE_FIRST_SUBJECT_ALL3_FIXED_EPOCH3_MODELS_BEFORE_IMAGE_EFFECT_READS',control_protocol=C.ref(a.protocol),training_protocol=q['training_protocol'],models=models,model_completions=finals,updates=7443,primary_exposures=119070,original_owner_retired=True,source_handoff=launch['handoff'],UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
        G.write(root/'first_subject_ready.json',ready);log('FIRST_SUBJECT_IMAGE_PHASE',models=models)
        jobpath=None;active='first_subject_postfit'
        cmd=[sys.executable,'-B',str(Path(q['image_protocol_path']).parent/'queue_HIVE_native_direction_postfit_CPU_20261103_v1.py'),'--protocol',q['image_protocol_path']]
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
        assert time.time()<q['whole_deadline_epoch']-q['maximum_image_phase_seconds']-120
        with (out/(active+'.stdout.log')).open('xb') as stdout,(out/(active+'.stderr.log')).open('xb') as stderr:
            worker=subprocess.Popen(cmd,cwd=Path(q['image_protocol_path']).parent,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
            C.atomic_json(out/'current.json',dict(PID=worker.pid,command=cmd,phase='first_subject_postfit'))
            rc=worker.wait(timeout=q['maximum_image_phase_seconds']+120)
        assert rc==0;worker=None
        imagespec=C.read(C.checked(launch['image_protocol']));donepath=Path(imagespec['output_root'])/'queue/completion.json';image=C.read(donepath)
        assert image['status']=='COMPLETE_FIXED_NATIVE_EEG_POSTFIT_STAGE_AND_INDEPENDENT_INPUT_SCORE_AUDITS' and image['protocol']==launch['image_protocol']
        decision=image['decision']['action']
        if decision!='PROMISING_MATCHED_INCREMENT_REQUIRES_MORE_SUBJECTS_AND_FULL_METRICS':
            assert decision in ('STOP_EXPANSION_JOINT_HARM','NO_CONFIRMED_MATCHED_INCREMENT_ON_THIS_EXPOSED_PANEL')
            G.write(out/'completion.json',dict(status='COMPLETE_FIRST_SUBJECT_MATCHED_IMAGES_REMAINING_SUBJECTS_NOT_EXPANDED_PENDING_LOG_REVIEW',protocol=C.ref(a.protocol),first_subject=C.ref(root/'first_subject_ready.json'),image_completion=C.ref(donepath),decision=image['decision'],original30model_plan_incomplete=True,remaining_models_not_declared_scientific_failures=True,whole_research_family_not_rejected=True,new_final_models=3,publication_ready=False,SOTA=False))
            log('FIRST_SUBJECT_COMPLETE_REVIEW_BEFORE_MORE_SUBJECTS',decision=decision);return
        log('MATCHED_IMAGE_SIGNAL_EXPAND_ORIGINAL_REMAINING_SUBJECTS',decision=decision)
        for subject in s['subjects'][1:]:
            for arm in C.ARMS:
                pin,weight=fit(subject,arm);finals.append(pin)
        assert len(finals)==30
        G.write(out/'completion.json',dict(status='COMPLETE_ALL30_ORIGINAL_RECIPE_FITS_AFTER_FIRST_SUBJECT_MATCHED_IMAGE_SIGNAL',protocol=C.ref(a.protocol),training_protocol=q['training_protocol'],models=finals,total_updates=74430,primary_exposures=1190700,first_subject_images=C.ref(donepath),multisubject_image_confirmation_still_required=True,publication_ready=False,SOTA=False))
        log('ALL30_FIXED_FITS_COMPLETE_MULTISUBJECT_IMAGE_CONFIRMATION_PENDING')
    except BaseException as error:
        if worker is not None and worker.poll() is None:
            if jobpath is not None and jobpath.exists():
                rec=C.read(jobpath);assert rec['wrapper_pid']==worker.pid;child=rec['pid']
                if Path('/proc',str(child)).exists():
                    assert b'train_HIVE_native_direction_full_T_GPU_20261103_v1.py' in Path('/proc',str(child),'cmdline').read_bytes()
                    os.killpg(child,signal.SIGTERM)
                    try:worker.wait(timeout=35)
                    except subprocess.TimeoutExpired:
                        if Path('/proc',str(child)).exists():os.killpg(child,signal.SIGKILL)
                        worker.wait(timeout=15)
            else:
                # The postfit coordinator owns its children and handles TERM.
                os.kill(worker.pid,signal.SIGTERM)
                try:worker.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    raise RuntimeError('Postfit coordinator did not exit after TERM; retain exact handle for inspection')
        G.write(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),active=active,no_automatic_retry=True,training_recipe_changed=False));raise


if __name__=='__main__':main()
