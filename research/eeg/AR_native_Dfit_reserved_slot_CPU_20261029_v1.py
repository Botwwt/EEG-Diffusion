"""Finite scheduling only: run the already-registered probe after native80.

The original CPU image waiter is suspended only after it has finished native
images/features and has no children. A separate bounded watchdog guarantees
SIGCONT if this controller disappears. No fitting/model/sampling files change.
"""
import argparse,datetime,hashlib,json,os,signal,subprocess,time,traceback
from pathlib import Path

M=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/multiscale_EEG_20261028_v1')
R=Path('/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1')
PY='/root/miniconda3/bin/python'
SPEC_SHA='77990e400ec0ade9862591b0ebdf1f3574c2c46cc045572db11e5eb184ca1f35'
WAITER=dict(pid=62214,start_ticks=1471417797)
JOB='AR_native_Dfit200_context_diagnostic_20261029_v1'
OUT=M/'AR_native_Dfit_reserved_slot_20261029_v1'

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def ref(path):
    with path.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(path),bytes=path.stat().st_size,sha256=h)
def proc(identity):
    try:f=(Path('/proc')/str(identity['pid'])/'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return None
    assert int(f[19])==identity['start_ticks'],'PID reused; refuse to signal'
    return f[0]
def children(pid):
    result=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdecimal():continue
        try:f=(p/'stat').read_text().rsplit(')',1)[1].split()
        except (FileNotFoundError,ProcessLookupError):continue
        if int(f[1])==pid:result.append(int(p.name))
    return result
def GPU_processes():
    return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
def resume_owned_waiter():
    state=proc(WAITER)
    if state in ('T','t'):os.kill(WAITER['pid'],signal.SIGCONT)
    return dict(UTC=now(),original_image_waiter=WAITER,state_before=state,resume_signal_sent=state in ('T','t'))
def watchdog():
    # Separate session, launched before SIGSTOP; crash-safe bounded reservation.
    write(OUT/'watchdog_started.json',dict(UTC=now(),pid=os.getpid(),maximum_hold_seconds=1100,original_image_waiter=WAITER))
    deadline=time.monotonic()+1100
    while time.monotonic()<deadline:
        if (OUT/'waiter_resumed.json').exists():return
        time.sleep(1)
    value=resume_owned_waiter();write(OUT/'watchdog_deadline_resume.json',value)

def run():
    assert not OUT.exists();OUT.mkdir()
    specpath=M/'AR_Dfit_context_code_v1/registered_native_Dfit200_spec_v1.json'
    assert ref(specpath)['sha256']==SPEC_SHA;spec=json.loads(specpath.read_bytes())
    producer=M/'AR_Dfit_context_code_v1/diagnose_AR_native_Dfit200_context_GPU_20261029_v1.py'
    for item in spec['source_pins']:
        assert ref(Path(item['path']))==item
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()=='GPU-62a1aee8-4740-dc59-93f4-d50644db650d'
    assert not (R/'audit'/f'{JOB}_job.json').exists()
    assert not Path(spec['output_root']).exists()
    launch=json.loads((M/'AR_anchored_images_code_v1/image_waiter_launch_new_v1.json').read_bytes())
    assert dict(pid=launch['pid'],start_ticks=launch['start_ticks'])==WAITER and proc(WAITER) not in (None,'Z','X','T','t')
    write(OUT/'registration.json',dict(status='ADMITTED_FINITE_SCHEDULING_OF_ALREADY_REGISTERED_FIXED_NATIVE_DIAGNOSTIC',UTC=now(),spec=ref(specpath),source=ref(Path(__file__)),original_image_waiter=WAITER,
        no_new_model_or_recipe=True,maximum_wait_seconds=21600,maximum_GPU_wrapper_seconds=900,maximum_waiter_hold_seconds=1100,
        scheduling='After native fixed final plus all native80 features; temporarily suspend only the childless CPU image waiter, resume on completion/failure or independent watchdog deadline. Training never suspended.'))
    started=time.monotonic();suspended=False;phase='WAIT_NATIVE_FIXED_FINAL_AND_NATIVE80_FEATURES';attempt=0
    def terminated(*_):raise InterruptedError('Reserved-slot controller explicitly terminated; preserve failure and resume owned CPU waiter')
    signal.signal(signal.SIGTERM,terminated)
    try:
        while True:
            assert time.monotonic()-started<21600
            train=M/'AR_anchored_training_20261029_v1';image=M/'AR_anchored_images_v1'
            assert not (train/'queue_failure.json').exists() and not (image/'controllers/new/failure.json').exists()
            required=[Path(spec['fixed_final_completion_path']),Path(spec['native_training_queue_completion']),Path(spec['native_stage80_artifact'])]
            required += [Path(spec['native_feature_root'])/f'{lo:05d}_{lo+20:05d}/completion.json' for lo in (0,20,40,60)]
            state=proc(WAITER);assert state not in (None,'Z','X')
            ready=all(p.is_file() for p in required)
            heartbeat=dict(status=phase,UTC=now(),ready=ready,image_waiter_state=state,seconds=time.monotonic()-started,new_optimizer_updates=0)
            pending=OUT/'progress.pending';pending.write_text(json.dumps(heartbeat,indent=2)+'\n');pending.replace(OUT/'progress.json')
            if not ready:time.sleep(20);continue
            assert json.loads(required[0].read_bytes())['updates']==8270
            assert json.loads(required[1].read_bytes())['status']=='ALL_HOST_ASSIGNED_FIXED_T_AR_ARMS_COMPLETE'
            wait=json.loads((image/'controllers/new/progress.json').read_bytes())
            if wait['status'] not in ('WAIT_FIXED_FINAL_paired_anchored','WAIT_EXACT_OLD_SCORE_OWNER_DECISION_COPY_NO_LOCAL_EFFECT_READ') or children(WAITER['pid']) or GPU_processes():
                time.sleep(10);continue
            phase='RESERVE_CHILDLESS_ORIGINAL_CPU_IMAGE_WAITER';attempt+=1
            assert proc(WAITER) not in (None,'Z','X','T','t')
            # Watchdog is armed before any suspension; never mutates fitting.
            if attempt==1:
                with (OUT/'watchdog.stdout.log').open('xb') as so,(OUT/'watchdog.stderr.log').open('xb') as se:
                    dog=subprocess.Popen([PY,'-B',str(Path(__file__)),'--watchdog'],stdin=subprocess.DEVNULL,stdout=so,stderr=se,start_new_session=True)
                deadline=time.monotonic()+5
                while not (OUT/'watchdog_started.json').exists():assert time.monotonic()<deadline;time.sleep(.05)
                write(OUT/'watchdog_launch.json',dict(UTC=now(),pid=dog.pid))
            os.kill(WAITER['pid'],signal.SIGSTOP);suspended=True
            deadline=time.monotonic()+3
            while proc(WAITER) not in ('T','t'):assert time.monotonic()<deadline;time.sleep(.02)
            if children(WAITER['pid']) or GPU_processes():
                value=resume_owned_waiter();suspended=False;write(OUT/f'reservation_race_{attempt:03d}_resumed.json',value)
                raise RuntimeError('Original waiter had started work during reservation; resumed immediately, no diagnostic launched')
            write(OUT/'waiter_suspended.json',dict(UTC=now(),original_image_waiter=WAITER,state=proc(WAITER),children=[],GPU_processes_empty=True,training_untouched=True))
            break
        phase='RUN_ONCE_REGISTERED_FIXED_FINAL_DIAGNOSTIC'
        command=[PY,'-B',str(R/'run_BrainHIVE_two_host_GPU_job_20261028_v1.py'),'--name',JOB,'--max-seconds','900','--',PY,'-B',str(producer),'--spec',str(specpath),'--job-name',JOB]
        write(OUT/'dispatch.json',dict(UTC=now(),command=command,spec=ref(specpath),new_optimizer_updates=0))
        with (OUT/'GPU.stdout.log').open('xb') as so,(OUT/'GPU.stderr.log').open('xb') as se:
            completed=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=so,stderr=se,timeout=970)
        assert completed.returncode==0,completed.returncode
        result=Path(spec['output_root'])/'completion.json';value=json.loads(result.read_bytes())
        assert value['status']=='COMPLETE_FIXED_NATIVE_FINAL_DFIT200_ALL_FIVE_CONTEXT_CONDITIONS_AND10SCALES' and value['queries']==200 and value['scales']==10
        write(OUT/'completion.json',dict(status='COMPLETE_ONCE_REGISTERED_NATIVE_DFIT200_PROBE_IN_RESERVED_SLOT',UTC=now(),result=ref(result),wrapper_receipt=ref(R/'audit'/f'{JOB}_job.json'),seconds=time.monotonic()-started,new_optimizer_updates=0,old_ordinary_untouched=True))
    except BaseException as error:
        write(OUT/'failure.json',dict(status='RESERVED_SLOT_FAILURE_PRESERVED_NO_RETRY_OR_RECIPE_CHANGE',UTC=now(),phase=phase,error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-started));raise
    finally:
        if suspended:
            value=resume_owned_waiter();write(OUT/'waiter_resumed.json',value)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watchdog',action='store_true');q=p.parse_args()
    watchdog() if q.watchdog else run()
