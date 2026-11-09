"""Finite new-job wrapper beside an identity-bound, fully stopped D host.

This explicit interleave uses TRIALSET_GPU_INTERLEAVE.lock, NOT GPU_JOB.lock.
The original D wrapper retains its lock and memory.  Only the new child may be
terminated here.  Total device memory protection is polled, not a reservation.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SCHEMA = 'BrainHIVE_paused_D_host_20261027_v1'
LOCK = ROOT / 'audit/TRIALSET_GPU_INTERLEAVE.lock'
MAX_TOTAL_BYTES = 30 << 30
DEAD = frozenset(('Z', 'X', 'x'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def atomic(path, value):
    path = Path(path)
    pending = path.with_name(path.name + '.pending')
    assert not pending.exists(), 'Preserve an unresolved receipt transaction'
    with pending.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def proc(pid):
    def status():
        value = Path('/proc', str(pid), 'stat').read_text()
        tail = value[value.rfind(')') + 2:].split()
        return dict(pid=int(pid), state=tail[0], ppid=int(tail[1]), pgid=int(tail[2]),
                    sid=int(tail[3]), start_ticks=int(tail[19]))
    try:
        first = status()
        argv = [os.fsdecode(part) for part in Path('/proc', str(pid), 'cmdline').read_bytes().split(b'\0') if part]
        second = status()
    except FileNotFoundError:
        return None
    assert first['start_ticks'] == second['start_ticks'], 'PID reused during inspection'
    return second | dict(argv=argv)


def bound_process(bound, *, stopped=False):
    current = proc(int(bound['pid']))
    assert current is not None and current['state'] not in DEAD, 'Bound host process disappeared'
    assert current['start_ticks'] == int(bound['start_ticks']) and current['argv'] == bound['argv'], 'Bound host identity changed'
    for key in ('pgid', 'sid'):
        if key in bound:
            assert current[key] == bound[key], 'Bound host process group changed'
    if stopped:
        assert current['state'] in ('T', 't'), 'D host must remain stopped throughout the new job'
    return current


def gpu_state():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                            check=True, capture_output=True, text=True, timeout=10)
    pids = {int(line.strip()) for line in result.stdout.splitlines() if line.strip()}
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                            check=True, capture_output=True, text=True, timeout=10)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 1, 'Existing single GPU only'
    used, total = [int(part.strip()) * (1 << 20) for part in lines[0].split(',')]
    return dict(pids=pids, used_bytes=used, total_bytes=total)


def validate_paused(proof_path, *, expected_sha=None, new_pid=None):
    proof_path = Path(proof_path).resolve()
    if expected_sha is not None:
        assert sha(proof_path) == expected_sha, 'Paused-host proof changed'
    proof = read(proof_path)
    assert proof['schema'] == SCHEMA
    registered = proof.get('processes', proof)
    identities = {role: bound_process(registered[role], stopped=True)
                  for role in ('controller', 'wrapper', 'trainer')}
    controller, wrapper, trainer = [identities[role] for role in ('controller', 'wrapper', 'trainer')]
    assert len({row['pid'] for row in identities.values()}) == 3
    assert trainer['ppid'] == wrapper['pid'] and wrapper['ppid'] == controller['pid']
    assert str(ROOT / 'queue_BrainHIVE_D_efficacy_focus_20261026_v1.py') in controller['argv']
    assert str(ROOT / 'run_gpu_job_v2.py') in wrapper['argv']
    assert str(ROOT / 'train_BrainHIVE_paired_effect_20261025_v1.py') in trainer['argv']
    assert '--resume' in trainer['argv'] and '--arm' in trainer['argv']
    assert trainer['argv'][trainer['argv'].index('--arm') + 1] == 'D'
    assert proof.get('GPU_trainer_pid', trainer['pid']) == trainer['pid']
    assert proof.get('maximum_total_GPU_used_bytes', MAX_TOTAL_BYTES) == MAX_TOTAL_BYTES
    job_name = 'BrainHIVE_paired_effect_v1_D_full_T_exact_resume_20261025'
    host_job = read(ROOT / 'audit' / (job_name + '_job.json'))
    assert host_job['name'] == job_name and host_job['status'] == 'RUNNING'
    assert host_job['pid'] == trainer['pid'] and host_job['wrapper_pid'] == wrapper['pid']
    assert host_job['command'] == trainer['argv']
    host_lock = ROOT / 'audit/GPU_JOB.lock'
    inode = host_lock.stat()
    assert any((fd.stat().st_dev, fd.stat().st_ino) == (inode.st_dev, inode.st_ino)
               for fd in Path('/proc', str(wrapper['pid']), 'fd').iterdir() if fd.exists())
    device = gpu_state()
    permitted = {trainer['pid']} | ({int(new_pid)} if new_pid is not None else set())
    assert trainer['pid'] in device['pids'], 'Stopped D must retain its original CUDA context'
    assert device['pids'] <= permitted, ('Unexpected GPU compute process', sorted(device['pids']))
    assert device['used_bytes'] <= MAX_TOTAL_BYTES, ('Total GPU usage exceeded 30 GiB; stop new child', device['used_bytes'], MAX_TOTAL_BYTES)
    return proof, identities, device


def verify_wrapper(document, pausedproof):
    """Child-side check for this explicit wrapper; never claim GPU_JOB ownership."""
    proof_path = Path(pausedproof).resolve()
    digest = sha(proof_path)
    parent = proc(os.getppid())
    assert parent is not None and parent['state'] not in DEAD
    command = parent['argv']
    assert str(ROOT / Path(__file__).name) in command and '--name' in command
    assert command[command.index('--name') + 1] == document['gpu_job_name']
    assert '--paused-host-proof' in command
    assert Path(command[command.index('--paused-host-proof') + 1]).resolve() == proof_path
    own = proc(os.getpid())
    assert own is not None and own['ppid'] == parent['pid']
    receipt = read(ROOT / 'audit' / (document['gpu_job_name'] + '_job.json'))
    assert receipt['status'] == 'RUNNING' and receipt['pid'] == os.getpid() and receipt['wrapper_pid'] == parent['pid']
    assert receipt['command'] == own['argv']
    assert receipt['wrapper_kind'] == 'PAUSED_D_HOST_EXPLICIT_INTERLEAVE'
    assert receipt['paused_host_proof'] == dict(path=str(proof_path), sha256=digest)
    assert 0 < receipt['max_seconds'] <= document['maximum_seconds'] <= 1800
    assert receipt['authorization_sha256'] == document['resource_authorization']['sha256']
    assert sha(receipt['authorization']) == receipt['authorization_sha256']
    assert receipt['lock_path'] == str(LOCK) and receipt['original_GPU_JOB_lock_owned'] is False
    inode = LOCK.stat()
    assert any((fd.stat().st_dev, fd.stat().st_ino) == (inode.st_dev, inode.st_ino)
               for fd in Path('/proc', str(parent['pid']), 'fd').iterdir() if fd.exists())
    with LOCK.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(lock, fcntl.LOCK_UN)
            raise AssertionError('New wrapper lacks its dedicated interleave lock')
    _, identities, device = validate_paused(proof_path, expected_sha=digest, new_pid=os.getpid())
    return dict(wrapper_pid=parent['pid'], job_name=document['gpu_job_name'],
                wrapper_kind=receipt['wrapper_kind'], paused_host_proof=receipt['paused_host_proof'],
                paused_D_trainer_pid=identities['trainer']['pid'], lock_path=str(LOCK),
                original_GPU_JOB_lock_owned=False, maximum_total_GPU_used_bytes=MAX_TOTAL_BYTES,
                total_GPU_used_at_validation=device['used_bytes'])


def stop_new_child(child, bound):
    """Finite cleanup of our separate-session child only."""
    sent = []
    for signum, seconds in ((signal.SIGTERM, 30), (signal.SIGKILL, 15)):
        if child.poll() is not None:
            break
        current = proc(child.pid)
        if current is not None and current['state'] not in DEAD:
            assert current['start_ticks'] == bound['start_ticks'] and current['argv'] == bound['argv']
            assert current['pgid'] == current['sid'] == child.pid
            os.killpg(child.pid, signum)
            sent.append(int(signum))
        try:
            child.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            continue
    assert child.poll() is not None, 'New child remained live; never signal the paused D host'
    return sent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--max-seconds', required=True, type=float)
    parser.add_argument('--paused-host-proof', required=True, type=Path)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == '--' else args.command
    assert command and args.name and '/' not in args.name and '\\' not in args.name
    assert math.isfinite(args.max_seconds) and 0 < args.max_seconds <= 1800
    proof_path = args.paused_host_proof.resolve()
    proof_sha = sha(proof_path)
    authorization = ROOT / 'configs/resource_authorization_20261020_v3.json'
    auth = read(authorization)
    assert auth['existing_single_GPU_only'] and auth['total_GPU_job_hours_cap'] is None
    assert not auth['purchase_or_resource_expansion_authorized']
    receipt_path = ROOT / 'audit' / (args.name + '_job.json')
    assert not receipt_path.exists() and not receipt_path.with_name(receipt_path.name + '.pending').exists()
    ledger = ROOT / 'audit/compute_ledger.jsonl'
    used = sum(float(json.loads(line).get('charged_seconds', 0))
               for line in ledger.read_text().splitlines() if line.strip())
    stopped = [False]
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, lambda number, frame: stopped.__setitem__(0, True))
    with LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _, identities, initial_device = validate_paused(proof_path, expected_sha=proof_sha)
        started = time.monotonic()
        child = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
        bound = proc(child.pid)
        assert bound is not None
        record = dict(name=args.name, pid=child.pid, wrapper_pid=os.getpid(), command=command,
            start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), max_seconds=args.max_seconds,
            charged_seconds_before_job=used, total_hours_cap=None, occupied_time_deadline=None,
            authorization=str(authorization), authorization_sha256=sha(authorization), status='RUNNING',
            wrapper_kind='PAUSED_D_HOST_EXPLICIT_INTERLEAVE', lock_path=str(LOCK),
            original_GPU_JOB_lock_owned=False, paused_host_proof=dict(path=str(proof_path), sha256=proof_sha),
            paused_D_identities=identities, maximum_total_GPU_used_bytes=MAX_TOTAL_BYTES,
            initial_total_GPU_used_bytes=initial_device['used_bytes'], memory_protection='0.5-second loop plus finite nvidia-smi latency; not a hard reservation')
        peak = initial_device['used_bytes']
        reason = None
        cleanup_signals = []
        try:
            atomic(receipt_path, record)
            while child.poll() is None:
                if stopped[0]:
                    reason = 'New wrapper termination requested; original D is never signaled'
                    break
                if time.monotonic() - started >= args.max_seconds:
                    reason = 'Finite registered job timeout'
                    break
                try:
                    _, _, device = validate_paused(proof_path, expected_sha=proof_sha, new_pid=child.pid)
                    peak = max(peak, device['used_bytes'])
                except BaseException as error:
                    reason = 'Paused host / GPU memory protection: ' + repr(error)
                    break
                time.sleep(.5)
        except BaseException as error:
            reason = 'Wrapper supervision failure: ' + repr(error)
        finally:
            if child.poll() is None:
                cleanup_signals = stop_new_child(child, bound)
            code = child.wait()
            elapsed = time.monotonic() - started
            record.update(status='COMPLETED' if code == 0 and reason is None else 'FAILED_OR_JOB_TIMEOUT_RESUMABLE',
                exit_code=code, charged_seconds=elapsed, end_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                peak_total_GPU_used_bytes=peak, new_child_cleanup_signals=cleanup_signals,
                paused_D_signals_sent=False)
            if reason is not None:
                record['termination_reason'] = reason
            atomic(receipt_path, record)
            with ledger.open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
        print(json.dumps(record), flush=True)
        return 0 if record['status'] == 'COMPLETED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
