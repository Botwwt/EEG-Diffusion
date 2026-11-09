"""Finite GPU interleave with identity-bound D checkpoint and automatic return."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OUT = Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/trialset_fastscreen_20261027_v1')
BASE = ROOT/'audit/trialset_fastscreen_20261027_v1'
DTRAIN = Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/paired_effect_20261025_v1/arms/D/train')


def read(p): return json.loads(Path(p).read_text())


def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def write(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name+'.pending'); t.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n'); t.replace(p)


def proc(pid):
    p = Path('/proc')/str(pid)
    try:
        fields = p.joinpath('stat').read_text().split(') ',1)[1].split()
        return dict(pid=pid, state=fields[0], ppid=int(fields[1]), start_ticks=int(fields[19]),
                    argv=p.joinpath('cmdline').read_bytes().rstrip(b'\0').decode().split('\0'))
    except FileNotFoundError: return None


def same(bound):
    current = proc(bound['pid'])
    assert current and current['start_ticks'] == bound['start_ticks'] and current['argv'] == bound['argv'], 'Process identity changed'
    return current


def alive(bound):
    current = proc(bound['pid'])
    if current is None or current['state'] == 'Z': return False
    same(bound); return True


def gpu_pids():
    r = subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                       check=True, capture_output=True, text=True, timeout=15)
    return [int(x.strip()) for x in r.stdout.splitlines() if x.strip()]


def can_return(control):
    active=gpu_pids()
    plan=read(control['plan_path'])
    for spec in plan['jobs']:
        path=ROOT/'audit'/(spec['name']+'_job.json')
        if not path.exists(): continue
        try: job=read(path)
        except json.JSONDecodeError: return False
        assert job['name']==spec['name'] and job['command']==spec['command']
        for pid in (job['pid'],job['wrapper_pid']):
            current=proc(pid)
            if current and current['state']!='Z': return False
    if active:
        return active==[control['trainer']['pid']] and alive(control['trainer'])
    # A CPU-initializing child may have no CUDA context yet; the shared lock
    # detects that launch/receipt window as well as unknown current GPU owners.
    with (ROOT/'audit/GPU_JOB.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return False
        fcntl.flock(lock,fcntl.LOCK_UN)
    return True


def checkpoint():
    import torch
    from train_BrainHIVE_bounded_component_v1 import seed
    torch.set_num_threads(2)
    state = torch.load(DTRAIN/'resume.pt', map_location='cpu', weights_only=False)
    progress = read(DTRAIN/'progress.json')
    assert progress['status'] == 'PAIRED_TRAIN_FINITE_STOP_EXACT_RESUME'
    assert state['cursor'] == state['seen'] == progress['cursor'] and state['updates'] == progress['updates']
    coverage = state['coverage']; assert tuple(coverage.shape) == (3,132300)
    assert int(coverage.sum()) == state['seen']
    provenance = read(DTRAIN/'input_provenance.json')
    for i in range(3):
        order = torch.randperm(132300, generator=torch.Generator().manual_seed(seed('pass'+str(i))))
        assert hashlib.sha256(order.numpy().tobytes()).hexdigest() == provenance['orders_sha256'][i]
        expected = torch.zeros(132300,dtype=torch.bool)
        expected[order[:max(0,min(132300,state['cursor']-i*132300))]] = True
        assert torch.equal(expected,coverage[i])
    assert sum(v.numel() for v in state['model'].values()) == 525568
    assert len(state['optimizer']['state']) > 0
    assert all(k in state for k in ('torch_rng','cuda_rng','noise_rng')) and len(state['cuda_rng']) == 1
    assert state['provenance_sha256'] == sha(DTRAIN/'input_provenance.json')
    payload = (DTRAIN/'training.jsonl').read_bytes()
    assert len(payload) == state['log_bytes'] and hashlib.sha256(payload).hexdigest() == state['log_sha256']
    assert not torch.cuda.is_initialized()
    receipt = dict(status='D_EXACT_CHECKPOINT_CPU_VERIFIED_FOR_SHORT_INTERLEAVE',
                   updates=state['updates'], primary_exposures=state['seen'], seconds=state['seconds'],
                   checkpoint_sha256=sha(DTRAIN/'resume.pt'), log_sha256=state['log_sha256'],
                   coverage_verified=True, optimizer_and_RNG_verified=True, CUDA_initialized=False)
    write(BASE/'D_checkpoint_before_screen.json', receipt)
    return receipt


def resume(control, reason):
    assert can_return(control), 'Pilot wrapper/child/lock must all close before returning GPU'
    if same(control['controller'])['state'] in ('T','t'):
        os.kill(control['controller']['pid'], signal.SIGCONT)
    write(BASE/'D_return_signal.json',dict(status='ORIGINAL_D_CONTROLLER_CONTINUED',reason=reason,
          controller=proc(control['controller']['pid']),UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))


def watchdog(path):
    control = read(path); deadline=time.monotonic()+control['watchdog_seconds']
    while time.monotonic()<deadline:
        helper=proc(control['supervisor']['pid'])
        if (BASE/'interleave_completion.json').exists(): return
        if helper is None or helper['state']=='Z' or helper['start_ticks']!=control['supervisor']['start_ticks'] or helper['argv']!=control['supervisor']['argv']:
            # Each new job is separately wrapped with a finite timeout. Wait for
            # that wrapper to close before returning the card to D.
            if can_return(control):
                resume(control,'watchdog: supervisor exited and GPU quiescent'); return
        time.sleep(3)
    # End the scheduler before cleanup so it cannot start the next pilot stage.
    helper=proc(control['supervisor']['pid'])
    if helper and helper['state']!='Z' and helper['start_ticks']==control['supervisor']['start_ticks'] and helper['argv']==control['supervisor']['argv']:
        same(control['supervisor']);os.kill(control['supervisor']['pid'],signal.SIGTERM)
    active_path=BASE/'active_new_job.json'
    if active_path.exists():
        active=read(active_path)
        plan=read(control['plan_path'])
        spec=next(j for j in plan['jobs'] if j['name']==active['name'])
        receipt_path=ROOT/'audit'/(active['name']+'_job.json')
        if receipt_path.exists():
            job=read(receipt_path)
            assert job['wrapper_pid']==active['wrapper_pid'] and job['command']==spec['command']
            child=proc(job['pid'])
            if child and child['state']!='Z':
                assert child['argv']==spec['command'] and child['ppid']==active['wrapper_pid']
                os.kill(child['pid'],signal.SIGTERM)
                until=time.monotonic()+35
                while alive(child) and time.monotonic()<until: time.sleep(1)
                if alive(child): same(child);os.kill(child['pid'],signal.SIGKILL)
            until=time.monotonic()+45
            while proc(active['wrapper_pid']) and proc(active['wrapper_pid'])['state']!='Z' and time.monotonic()<until: time.sleep(1)
    if can_return(control): resume(control,'watchdog: finite deadline, owned pilot cleanup complete')
    else: write(BASE/'watchdog_attention.json',dict(status='UNVERIFIED_GPU_OWNER_REQUIRES_MANUAL_CHECK',pids=gpu_pids()))


def main(plan_path):
    BASE.mkdir(parents=True, exist_ok=True)
    assert not (BASE/'control.json').exists(), 'Single attempt; preserve earlier control evidence'
    plan=read(plan_path)
    assert read(OUT/'inputs/completion.json')['status']=='PHYSICAL_R4_T_INPUTS_READY'
    assert sha(OUT/'panel.json')==plan['panel_sha256']
    for name,digest in plan['source_sha256'].items(): assert sha(ROOT/name)==digest,name
    queue=read(ROOT/'audit/efficacy_focus_20261026_v1/queue_state.json')
    original_job=ROOT/'audit/BrainHIVE_paired_effect_v1_D_full_T_20261025_job.json'
    job=read(original_job)
    assert queue['status']=='D_ONLY_FULL_T_FULL_V_EFFICACY_RUNNING' and queue['pid']==163400
    assert queue['current_stage']=='BrainHIVE_paired_effect_v1_D_full_T_20261025'
    assert queue['child_pid']==job['wrapper_pid']==163413 and job['pid']==163414 and job['status']=='RUNNING'
    assert not (ROOT/'audit/BrainHIVE_paired_effect_v1_D_full_T_exact_resume_20261025_job.json').exists()
    assert not (ROOT/'audit/efficacy_focus_20261026_v1/BrainHIVE_paired_effect_v1_D_full_T_exact_resume_20261025.log').exists()
    controller, wrapper, trainer = [proc(p) for p in (163400,163413,163414)]
    assert str(ROOT/'queue_BrainHIVE_D_efficacy_focus_20261026_v1.py') in controller['argv']
    assert str(ROOT/'run_gpu_job_v2.py') in wrapper['argv']
    assert trainer['argv']==job['command'] and trainer['ppid']==wrapper['pid']
    assert gpu_pids()==[trainer['pid']]
    control=dict(controller=controller,wrapper=wrapper,trainer=trainer,supervisor=proc(os.getpid()),
                 original_job_sha256=sha(original_job),plan_sha256=sha(plan_path),plan_path=str(Path(plan_path).resolve()),
                 watchdog_seconds=sum(j['max_seconds']+90 for j in plan['jobs'])+600)
    write(BASE/'control.json',control)
    guard=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--watchdog',str(BASE/'control.json')],
                           stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    stopped=False
    results=[]
    try:
        same(controller);os.kill(controller['pid'],signal.SIGSTOP);stopped=True
        time.sleep(.2);assert same(controller)['state'] in ('T','t')
        same(trainer);os.kill(trainer['pid'],signal.SIGTERM)
        until=time.monotonic()+100
        while alive(trainer) or alive(wrapper):
            assert time.monotonic()<until,'D trainer/wrapper did not close within the bounded interval'
            time.sleep(1)
        assert not gpu_pids()
        closed=read(original_job)
        assert closed['status']=='FAILED_OR_JOB_TIMEOUT_RESUMABLE' and closed['exit_code']!=0
        for key in ('name','command','wrapper_pid','pid'): assert closed[key]==job[key]
        assert closed['charged_seconds']>0 and closed['end_utc']
        ledger=[json.loads(line) for line in (ROOT/'audit/compute_ledger.jsonl').read_text().splitlines()]
        assert any(row==closed for row in ledger), 'Original D wrapper ledger not closed'
        evidence=checkpoint()
        print(json.dumps(evidence),flush=True)
        for spec in plan['jobs']:
            assert not gpu_pids()
            for name,digest in plan['source_sha256'].items(): assert sha(ROOT/name)==digest
            command=[sys.executable,'-B',str(ROOT/'run_gpu_job_v2.py'),'--name',spec['name'],
                     '--max-seconds',str(spec['max_seconds']),'--',*spec['command']]
            with (OUT/(spec['stage']+'.log')).open('xb') as stream:
                p=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                                   start_new_session=True,env=os.environ.copy())
                write(BASE/'active_new_job.json',dict(stage=spec['stage'],wrapper_pid=p.pid,name=spec['name']))
                # The wrapper implements its own TERM/KILL finite deadline.
                code=p.wait(timeout=spec['max_seconds']+90)
            receipt=read(ROOT/'audit'/(spec['name']+'_job.json'))
            assert not gpu_pids()
            results.append(dict(stage=spec['stage'],exit_code=code,receipt=receipt))
            write(BASE/'new_jobs.json',results)
            assert code==0 and receipt['status']=='COMPLETED', 'Failed attempt preserved; no adaptive replay'
        write(BASE/'interleave_results.json',dict(status='NEW_FASTSCREEN_JOBS_COMPLETE',jobs=results))
    except BaseException as error:
        write(BASE/'interleave_failure.json',dict(error=repr(error),jobs=results))
        raise
    finally:
        if stopped and can_return(control):
            resume(control,'short GPU screen finished or failed; return preserved D checkpoint')
            write(BASE/'interleave_completion.json',dict(status='D_CONTROLLER_CONTINUE_SIGNAL_SENT',
                  actual_D_resumed_checkpoint_verification='pending separate read-only observation',jobs=results))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan');parser.add_argument('--watchdog')
    args=parser.parse_args()
    if args.watchdog: watchdog(args.watchdog)
    else: main(args.plan)
