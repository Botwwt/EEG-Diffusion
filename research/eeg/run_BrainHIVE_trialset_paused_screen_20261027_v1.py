"""Pause D in memory for one finite screen, then continue the same processes.

The earlier device-index engineering failure and checkpoint-resume evidence are
preserved. No D retry slot or scientific update is added by this scheduler.
"""
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

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'audit/trialset_fastscreen_20261027_v1/pause_v2'
OUT=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/trialset_fastscreen_20261027_v1')


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.pending');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(p)


def proc(pid):
    p=Path('/proc')/str(pid)
    try:
        f=p.joinpath('stat').read_text().split(') ',1)[1].split()
        return dict(pid=pid,state=f[0],ppid=int(f[1]),start_ticks=int(f[19]),argv=p.joinpath('cmdline').read_bytes().rstrip(b'\0').decode().split('\0'))
    except FileNotFoundError:return None


def live(p):return p is not None and p['state']!='Z'


def same(bound):
    p=proc(bound['pid']);assert live(p) and p['start_ticks']==bound['start_ticks'] and p['argv']==bound['argv'],'Process identity changed'
    return p


def gpu():
    r=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
    return {int(x) for x in r.stdout.splitlines() if x.strip()}


def quiescent(proof,lock_held=False):
    # Cover wrappers before their receipt/lock/CUDA initialization. Exact argv
    # contains both our dedicated wrapper and a unique registered job name.
    names={j['name'] for j in proof['jobs']}
    wrapper=str(ROOT/'run_gpu_job_BrainHIVE_paused_host_20261027_v1.py')
    for path in Path('/proc').glob('[0-9]*'):
        p=proc(int(path.name))
        if not live(p) or wrapper not in p['argv'] or '--name' not in p['argv']:continue
        pos=p['argv'].index('--name')+1
        if pos<len(p['argv']) and p['argv'][pos] in names:return False
    for j in proof['jobs']:
        p=ROOT/'audit'/(j['name']+'_job.json')
        if not p.exists():continue
        try:r=read(p)
        except json.JSONDecodeError:return False
        assert r['command']==j['command'] and r['name']==j['name']
        if live(proc(r['pid'])) or live(proc(r['wrapper_pid'])):return False
    if not gpu()<={proof['GPU_trainer_pid']}:return False
    if lock_held:return True
    with (ROOT/'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return False
        fcntl.flock(f,fcntl.LOCK_UN)
    return True


def stop_new(proof):
    for j in proof['jobs']:
        path=ROOT/'audit'/(j['name']+'_job.json')
        if not path.exists():continue
        r=read(path);assert r['command']==j['command'] and r['name']==j['name']
        p=proc(r['pid'])
        if live(p):
            assert p['argv']==j['command'] and p['ppid']==r['wrapper_pid']
            os.kill(p['pid'],signal.SIGTERM)
            until=time.monotonic()+35
            while live(proc(p['pid'])) and time.monotonic()<until:time.sleep(1)
            if live(proc(p['pid'])):same(p);os.kill(p['pid'],signal.SIGKILL)
        until=time.monotonic()+45
        while live(proc(r['wrapper_pid'])) and time.monotonic()<until:time.sleep(1)


def return_D(proof,reason):
    with (ROOT/'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert quiescent(proof,lock_held=True),'New wrappers and children must close'
        for role in ('trainer','wrapper','controller'):
            p=same(proof['processes'][role])
            if p['state'] in ('T','t'):os.kill(p['pid'],signal.SIGCONT)
        time.sleep(.2)
        observed={role:same(p) for role,p in proof['processes'].items()}
        assert all(p['state'] not in ('T','t') for p in observed.values())
        # A late wrapper acquires this lock only after D has continued; its own
        # under-lock STOP-state validation will then reject before child launch.
        fcntl.flock(lock,fcntl.LOCK_UN)
    write(BASE/'return_signal.json',dict(status='SAME_D_PROCESSES_CONTINUED',reason=reason,processes=observed,UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))


def guard(path):
    proof=read(path);deadline=time.monotonic()+sum(j['max_seconds']+90 for j in proof['jobs'])+300
    while time.monotonic()<deadline:
        if (BASE/'completion.json').exists():return
        p=proc(proof['supervisor']['pid']);bound=proof['supervisor']
        if not live(p) or p['start_ticks']!=bound['start_ticks'] or p['argv']!=bound['argv']:
            # CPU initialization can precede both a receipt and a CUDA context.
            # Wait for the unique finite wrapper to publish/close before return.
            time.sleep(3)
            if quiescent(proof):return_D(proof,'watchdog: supervisor exited');return
        time.sleep(2)
    p=proc(proof['supervisor']['pid'])
    if live(p) and p['start_ticks']==proof['supervisor']['start_ticks'] and p['argv']==proof['supervisor']['argv']:
        os.kill(p['pid'],signal.SIGTERM)
    stop_new(proof)
    return_D(proof,'watchdog: bounded screen deadline and owned cleanup')


def main(plan_path):
    BASE.mkdir(parents=True,exist_ok=True);assert not (BASE/'proof.json').exists()
    plan=read(plan_path)
    for name,h in plan['source_sha256'].items():assert sha(ROOT/name)==h,name
    queue=read(ROOT/'audit/efficacy_focus_20261026_v1/queue_state.json')
    host_job=read(ROOT/'audit/BrainHIVE_paired_effect_v1_D_full_T_exact_resume_20261025_job.json')
    assert queue['status']=='D_ONLY_FULL_T_FULL_V_EFFICACY_RUNNING' and queue['pid']==163400
    assert queue['child_pid']==host_job['wrapper_pid'] and host_job['status']=='RUNNING'
    processes=dict(controller=proc(queue['pid']),wrapper=proc(host_job['wrapper_pid']),trainer=proc(host_job['pid']))
    assert processes['trainer']['argv']==host_job['command'] and processes['trainer']['ppid']==processes['wrapper']['pid']
    assert str(ROOT/'queue_BrainHIVE_D_efficacy_focus_20261026_v1.py') in processes['controller']['argv']
    assert gpu()=={processes['trainer']['pid']}
    proof=dict(schema='BrainHIVE_paused_D_host_20261027_v1',processes=processes,
               GPU_trainer_pid=processes['trainer']['pid'],maximum_total_GPU_used_bytes=30<<30,
               jobs=plan['jobs'],supervisor=proc(os.getpid()),plan_sha256=sha(plan_path),
               source_sha256=plan['source_sha256'],D_resume_slot_unchanged=True)
    write(BASE/'proof.json',proof)
    subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--watchdog',str(BASE/'proof.json')],
                     stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    results=[];paused=False
    try:
        for role in ('controller','wrapper','trainer'):
            same(processes[role]);os.kill(processes[role]['pid'],signal.SIGSTOP);paused=True
        time.sleep(2)
        assert all(same(p)['state'] in ('T','t') for p in processes.values())
        print(json.dumps(dict(status='D_PAUSED_IN_MEMORY_NO_RESTART',processes=processes)),flush=True)
        for j in plan['jobs']:
            for name,h in plan['source_sha256'].items():assert sha(ROOT/name)==h,name
            command=[sys.executable,'-B',str(ROOT/'run_gpu_job_BrainHIVE_paused_host_20261027_v1.py'),
                     '--name',j['name'],'--max-seconds',str(j['max_seconds']),'--paused-host-proof',str(BASE/'proof.json'),'--',*j['command']]
            with (OUT/(j['stage']+'_runtimefix.log')).open('xb') as f:
                p=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                code=p.wait(timeout=j['max_seconds']+100)
            r=read(ROOT/'audit'/(j['name']+'_job.json'))
            results.append(dict(stage=j['stage'],exit_code=code,receipt=r));write(BASE/'jobs.json',results)
            assert code==0 and r['status']=='COMPLETED','Engineering failure preserved'
        write(BASE/'results.json',dict(status='FASTSCREEN_FIT_IMAGES_SCORE_COMPLETE',jobs=results))
    except BaseException as e:
        write(BASE/'failure.json',dict(error=repr(e),jobs=results));raise
    finally:
        if paused:
            if not quiescent(proof):stop_new(proof)
            return_D(proof,'finite screen finished/failed; continue original D without a restart')
            write(BASE/'completion.json',dict(status='D_CONTINUE_SIGNAL_SENT',jobs=results))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--plan');a.add_argument('--watchdog');args=a.parse_args()
    if args.watchdog:
        try:guard(args.watchdog)
        except BaseException as e:
            write(BASE/'watchdog_attention.json',dict(status='RECOVERY_REQUIRES_ROOT_OBSERVATION',error=repr(e)));raise
    else:main(args.plan)
