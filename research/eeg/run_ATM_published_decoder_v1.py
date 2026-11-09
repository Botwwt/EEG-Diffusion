"""Finite serial decoder replication after the current complete-score closure."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--registration',type=Path,required=True)
    p.add_argument('--registration-sha256',required=True)
    a=p.parse_args()
    assert sha(a.registration)==a.registration_sha256
    cfg=json.loads(a.registration.read_text())
    root=Path(__file__).resolve().parent
    out=Path(cfg['output_root'])
    state_path=root/'audit/ATM_published_decoder_state_20261020_v1.json'
    assert not state_path.exists(), 'A separately named retry needs a new registration'
    lock=(root/'audit/ATM_published_decoder_CONTROLLER.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(status='WAITING_FOR_COMPLETE_FROZEN_SCORE_CLOSURE',controller_pid=os.getpid(),
               registration=str(a.registration),registration_sha256=a.registration_sha256,
               phase=None,completed_phases=[],scientific_goal_complete=False,new_component_training=False)

    def save():
        state['UTC']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        temp=state_path.with_suffix('.tmp')
        temp.write_text(json.dumps(state,indent=2)+'\n')
        os.replace(temp,state_path)

    save()
    caps=cfg['phase_caps_seconds']
    deadline=json.loads((root/'configs/compute_time_bounds.json').read_text())['conservative_occupied_time_deadline_epoch']
    reservation=sum(caps.values())+cfg['reserve_seconds']+45
    closure=Path(cfg['required_prior_closure'])
    while not closure.exists():
        if time.time()+reservation>=deadline:
            state.update(status='STOPPED_FULL_LIFECYCLE_UNFUNDED_BEFORE_NEW_GPU',reason='Remaining occupied-time deadline cannot fund all registered phases and reserve')
            save()
            return 0
        time.sleep(30)
    assert json.loads(closure.read_text())['status']==cfg['required_prior_closure_status']
    try:
        for path,digest in cfg['source_sha256'].items():
            assert sha(path)==digest,path
        used=sum(float(x.get('charged_seconds',0)) for x in map(json.loads,(root/'audit/compute_ledger.jsonl').read_text().splitlines()))
        assert used+reservation<40*3600 and time.time()+reservation<deadline
        import shutil
        assert shutil.disk_usage('/root').free>2000*1024**2+300*1024**2+1024**3
        current=int(Path('/sys/fs/cgroup/memory.current').read_text())
        maximum=int(Path('/sys/fs/cgroup/memory.max').read_text())
        stats={k:int(v) for k,v in (line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())}
        reclaimable=min(stats.get('inactive_file',0),8*1024**3)
        assert maximum-current+reclaimable>=16*1024**3,'Conservative loader memory reservation unavailable'
        budget=dict(status='FUNDED_COMPLETE_PUBLIC_DECODER_LIFECYCLE',occupied_deadline_epoch=deadline,
                    reserved_seconds=reservation,charged_seconds_before_start=used,
                    root_free_bytes=shutil.disk_usage('/root').free,memory_current=current,
                    memory_max=maximum,conservatively_reclaimable_inactive_file=reclaimable,
                    prior_closure_sha256=sha(closure),registration_sha256=a.registration_sha256)
        (out/'budget_before_start.json').write_text(json.dumps(budget,indent=2)+'\n')
        logs=out/'logs'
        logs.mkdir()
        env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TQDM_DISABLE='1',
                 OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONPATH=':'.join(cfg['python_paths']))
        base=[sys.executable]
        reg=['--registration',str(a.registration)]
        phases=[
            ('generate',True,base+[str(root/'generate_ATM_published_decoder_v1.py')]+reg),
            ('join',False,base+[str(root/'audit_ATM_published_decoder_v1.py')]+reg+['--mode','join']),
            ('gallery_features',True,base+[str(root/'extract_image_features.py'),'--manifest',cfg['gallery_manifest'],'--output',str(out/'gallery/features'),'--real-gallery']),
            ('features',True,base+[str(root/'extract_image_features.py'),'--manifest',str(out/'manifest.jsonl'),'--output',str(out/'features')]),
            ('score',False,base+[str(root/'score_full_manifest.py'),'--manifest',str(out/'manifest.jsonl'),'--features',str(out/'features'),'--gallery-manifest',cfg['gallery_manifest'],'--gallery-features',str(out/'gallery/features'),'--output',str(out/'scores')]),
            ('lowlevel',False,base+[str(root/'score_low_level.py'),'--manifest',str(out/'manifest.jsonl'),'--gallery-manifest',cfg['gallery_manifest'],'--output',str(out/'lowlevel')]),
            ('verify',False,base+[str(root/'audit_ATM_published_decoder_v1.py')]+reg+['--mode','verify']),
            ('report',False,base+[str(root/'audit_ATM_published_decoder_v1.py')]+reg+['--mode','report']),
        ]
        for index,(phase,gpu,command) in enumerate(phases):
            remaining=sum(caps[x[0]] for x in phases[index:])+cfg['reserve_seconds']+45
            assert time.time()+remaining<deadline,'Remaining full lifecycle no longer fits the occupied-time deadline'
            actual=(base+[str(root/'run_gpu_job.py'),'--name','ATM_public_decoder_'+phase+'_20261020_v1',
                          '--max-seconds',str(caps[phase]),'--']+command) if gpu else command
            phase_env=env if gpu else env|{'CUDA_VISIBLE_DEVICES':''}
            started=time.monotonic()
            with (logs/(phase+'.log')).open('x') as log:
                child=subprocess.Popen(actual,cwd=root,env=phase_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state.update(status='RUNNING',phase=phase,phase_supervisor_pid=child.pid,command=actual)
                save()
                try:
                    code=child.wait(timeout=caps[phase]+90 if gpu else caps[phase])
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:
                        code=child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL)
                        code=child.wait()
            elapsed=time.monotonic()-started
            if not gpu:
                record=dict(event_id='ATM_public_decoder_'+phase+'_20261020_v1',seconds=elapsed,
                            GPU_used=False,execution_location='server_CPU',exit_code=code,
                            command=command,registration_sha256=a.registration_sha256)
                with (root/'audit/CPU_postprocessing_ledger.jsonl').open('a') as f:
                    f.write(json.dumps(record)+'\n')
            assert code==0,(phase,code,str(logs/(phase+'.log')))
            state['completed_phases'].append(phase)
            state['phase_supervisor_pid']=None
            save()
        state.update(status='COMPLETED_PUBLIC_DECODER_BRANCH_ALL2000_NOT_FULL_EEG_REPRODUCTION',
                     report_completion=str(out/'report_completion.json'))
        save()
    except Exception:
        state.update(status='STOPPED_PRESERVED_INCOMPLETE_PUBLIC_DECODER_LIFECYCLE',error=traceback.format_exc())
        save()
        raise


if __name__=='__main__':
    main()
