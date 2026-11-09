"""Finite four-arm D39 full-T training controller, original ordinary immutable.

This controller dispatches only training and CPU checkpoint audits. Image
evaluation has a separately locked plan and is not declared completed here.
"""
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def checked(x):
    p=Path(x['path']);assert p.is_file() and sha(p)==x['sha256']
    if 'bytes' in x:assert p.stat().st_size==x['bytes']
    return p
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def atomic(p,x):
    t=p.with_name(p.name+'.pending');assert not t.exists()
    with t.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(t,p)

def run(sp):
    started=time.monotonic();spec=read(sp);assert spec['status']=='D39_FOUR_ARM_NATIVE_CORE_ONE_COMPLETE_T_PASS_LOCKED'
    for x in spec['source_pins']:checked(x)
    B=Path(spec['output']);Q=B/'training_queue_v1';assert not Q.exists();Q.mkdir(parents=True)
    completed=[];segment_number=0;stage='START'
    try:
        for arm in spec['arms']:
            arm_dir=B/'arms'/arm;arm_dir.mkdir(parents=True,exist_ok=True)
            while not (arm_dir/'completion.json').exists():
                assert time.monotonic()-started<spec['maximum_controller_seconds'] and segment_number<64
                for x in spec['source_pins']:checked(x)
                current=read(arm_dir/'latest_commit.json') if (arm_dir/'latest_commit.json').exists() else None
                updates=current['updates'] if current else 0
                if current:
                    resume=arm_dir/f'checkpoint_{updates:06d}.json';assert read(resume)==current;checked(current['checkpoint'])
                else:resume=None
                assert updates<827
                assert shutil.disk_usage(B).free>spec['remaining_disk_buffer_bytes']
                job=f'D39_core_{arm}_{updates:06d}_20261028_v1';segment=Q/(job+'_segment.json');segment_number+=1
                write(segment,dict(arm=arm,GPU_job_name=job,start_updates=updates,resume_commit=ref(resume) if resume else None,
                    training_spec_sha256=sha(sp),maximum_updates=256,unique_T_queries=13230))
                command=[sys.executable,'-B',str(ROOT/'run_BrainHIVE_two_host_GPU_job_20261028_v1.py'),'--name',job,'--max-seconds','900','--',
                    sys.executable,'-B',str(ROOT/'train_BrainHIVE_measurement_prior_GPU_20261028_v1.py'),'--spec',str(sp),'--arm',arm,'--gpu-job-name',job,'--segment',str(segment)]
                active=dict(status='D39_FOUR_ARM_FULL_T_TRAINING_QUEUE_RUNNING',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),arm=arm,
                    start_updates=updates,GPU_job_name=job,segments=segment_number,completed_segments=completed,command=command)
                atomic(Q/'progress.json',active);stage=job
                with (Q/(job+'.stdout.log')).open('xb') as out,(Q/(job+'.stderr.log')).open('xb') as err:
                    p=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err)
                    active['wrapper_pid']=p.pid;atomic(Q/'progress.json',active)
                    rc=p.wait(timeout=945)
                receipt=read(ROOT/'audit'/(job+'_job.json'));assert rc==0 and receipt['exit_code']==0,(rc,receipt)
                finish=arm_dir/(job+'_segment_completion.json');segment_done=read(finish)
                assert segment_done['status']=='D39_NATIVE_CORE_SEGMENT_COMMITTED_AND_FROZEN_HOST_VERIFIED' and segment_done['end_updates']>updates
                # Validate every naturally saved64-update checkpoint from this
                # segment, without pausing or selecting a scientific checkpoint.
                for cp in sorted(arm_dir.glob('checkpoint_*.json')):
                    value=read(cp)
                    if value['updates']<=updates:continue
                    audit=Q/f'{arm}_{value["updates"]:06d}_checkpoint_CPU_audit.json'
                    cmd=[sys.executable,'-B',str(ROOT/'audit_BrainHIVE_measurement_checkpoint_CPU_20261028_v1.py'),'--spec',str(sp),'--commit',str(cp),'--output',str(audit)]
                    with audit.with_suffix('.stdout.log').open('xb') as out,audit.with_suffix('.stderr.log').open('xb') as err:r=subprocess.run(cmd,cwd=ROOT,stdout=out,stderr=err,timeout=180)
                    assert r.returncode==0,(cmd,r.returncode)
                    assert read(audit)['status']=='D39_REAL_COMMITTED_MODEL_ADAMW_RNG_COVERAGE_AND_LOG_PREFIX_CPU_VERIFIED'
                completed.append(dict(stage=job,exit_code=rc,segment_completion=ref(finish)))
            done=read(arm_dir/'completion.json');assert done['updates']==827 and done['query_exposures']==13230;checked(done['model'])
        assert all(read(B/'arms'/a/'completion.json')['fixed_final_not_held_selected'] for a in spec['arms'])
        write(Q/'completion.json',dict(status='D39_ALL4_NATIVE_CORE_COMPLETE_T_ONE_PASS_AND_CPU_COMMITS_TRAINING_CLOSED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            spec=ref(sp),arms=spec['arms'],segments=segment_number,steps=completed,
            fits=[ref(B/'arms'/a/'completion.json') for a in spec['arms']],native_core_models=4,total_updates=3308,total_T_query_exposures=52920,
            original_ordinary_unchanged=True,research_images_generated=0,image_efficacy_established=False,
            image_protocol=spec['locked_future_image_policy'],seconds=time.monotonic()-started))
    except BaseException as e:
        write(Q/'failure.json',dict(status='D39_TRAINING_QUEUE_FAILURE_PRESERVED_NO_AUTOMATIC_RETRY',stage=stage,error=repr(e),traceback=traceback.format_exc(),segments=segment_number,
           completed=completed,seconds=time.monotonic()-started));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);run(p.parse_args().spec)
