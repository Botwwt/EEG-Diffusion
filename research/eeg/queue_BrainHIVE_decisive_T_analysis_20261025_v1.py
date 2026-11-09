"""Finite, independently sealed T-only analysis after the complete four-arm run.

No Torch import, training, score-table access, changes to frozen campaign
sources, cache reconstruction, or automatic retry. Waiting reads only bounded
JSON/source files and /proc. Each child owns its process group; the GPU child
is stopped only through its matching wrapper receipt.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SERVER = Path('/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1')
WORK = Path('/root/brainhive_reproduction_20261020_v1')
DATA = Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')
BASE = ROOT / 'audit/decisive_T_analysis_20261025_v1'
STATE = BASE / 'queue_state.json'
UPSTREAM_BASE = ROOT / 'audit/functional_full_lifecycle_20261022_v1'
SCHEMA = 'BrainHIVE_decisive_T_analysis_queue_20261025_v1'
SUCCESS = 'ALL_FOUR_FULL_LIFECYCLES_COMPLETE_AFTER_DOCUMENTED_SINGLE_O_RESTART'
COMPARISON = 'ALL_FOUR_FULL16600_V_PAIRED_COMPARISONS_COMPLETE'
STAGES = (
    ('logsCPU', 'analyze_BrainHIVE_functional_T_logs_CPU_20261025_v1.py', False),
    ('layersCPU', 'analyze_BrainHIVE_EEG_layers_T_CPU_20261025_v1.py', False),
    ('functionGPU', 'diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v1.py', True),
    ('directionCPU', 'read_BrainHIVE_decisive_T_analysis_CPU_20261025_v1.py', False),
)
STAGE_SECONDS = dict(logsCPU=1800, layersCPU=14400, functionGPU=7200, directionCPU=1800)
LIMITS = dict(wait_seconds=172800, poll_seconds=30,
    maximum_output_bytes=19 << 28, minimum_disk_extra_bytes=5 << 30,
    minimum_metadata_disk_extra_bytes=1 << 30,
    minimum_CPU_headroom_bytes=10 << 30, minimum_free_GPU_bytes=20 << 30)
JSON_CAP = 16 << 20
SOURCE_CAP = 4 << 20
LOG_CAP = 16 << 20
METADATA_CAP = 64 << 20
ARMS = ['O', 'P', 'PN', 'S']


def sha(path, maximum=None):
    path = Path(path)
    if maximum is not None:
        assert path.stat().st_size <= maximum, ('Bounded input exceeded', str(path))
    before = path.stat()
    with path.open('rb') as stream:
        value = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    assert (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'Input changed during hash'
    return value


def read(path):
    path = Path(path)
    assert path.stat().st_size <= JSON_CAP, ('Bounded JSON exceeded', str(path))
    return json.loads(path.read_bytes())


def ref(path):
    path = Path(path)
    return dict(path=str(path), sha256=sha(path, JSON_CAP), bytes=path.stat().st_size)


def read_ref(item):
    assert isinstance(item, dict) and set(item) >= {'path', 'sha256'}
    assert sha(item['path'], JSON_CAP) == item['sha256'], item['path']
    if 'bytes' in item:
        assert Path(item['path']).stat().st_size == item['bytes']
    return read(item['path'])


def atomic(path, value):
    value['UTC'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    pending = Path(path).with_suffix('.pending')
    payload = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    assert len(payload) <= JSON_CAP
    pending.write_bytes(payload)
    pending.replace(path)


def live(pid):
    if not pid:
        return False
    try:
        value = Path('/proc', str(int(pid)), 'stat').read_text()
        return value[value.rfind(')') + 2:].split()[0] not in ('Z', 'X')
    except (OSError, ValueError, IndexError):
        return False


def core_sha(recipe):
    core = {key: value for key, value in recipe.items()
        if key not in {'admission_receipt', 'admission_sha256'}}
    return hashlib.sha256(json.dumps(core, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_registration(value, *, root=ROOT, check_files=True, enforce_paths=True):
    assert value['schema'] == SCHEMA
    assert value['limits'] == LIMITS, 'Fixed finite resource limits must match'
    assert value.get('auto_fit', False) is False
    assert value.get('new_candidate_training', False) is False
    assert len(value['stages']) == len(STAGES)
    names, completions = set(), set()
    roots = [Path(p) for p in value['output_roots']]
    assert len(roots) == 2 and len(set(roots)) == 2
    assert all(path.is_absolute() and '..' not in path.parts for path in roots)
    assert not any(a != b and a.is_relative_to(b) for a in roots for b in roots)
    if enforce_paths:
        assert set(roots) == {DATA / 'analysis/decisive_T_analysis_20261025_v1', BASE}
        upstream = value['upstream']
        assert Path(upstream['state_path']) == UPSTREAM_BASE / 'formal_recovery_queue.json'
        assert Path(upstream['lifecycle_lock']) == UPSTREAM_BASE / 'lifecycle_controller.lock'
        assert Path(upstream['controller']['path']) == ROOT / 'queue_BrainHIVE_functional_recovery_20261022_v1.py'
        assert Path(value['wrapper']['path']) == ROOT / 'run_gpu_job_v2.py'
        assert Path(value['resource_helper']['path']) == ROOT / 'brainhive_functional_training_contract_v2.py'
        assert Path(value['authorization']['path']) == ROOT / 'configs/resource_authorization_20261020_v3.json'
    pins = value['source_sha256']
    required = {Path(__file__).name, 'run_gpu_job_v2.py', 'brainhive_functional_training_contract_v2.py',
        *(entry for _, entry, _ in STAGES)}
    assert required <= set(pins)
    for name, digest in pins.items():
        assert Path(name).name == name and len(digest) == 64
        if check_files:
            assert sha(root / name, SOURCE_CAP) == digest, name
    for stage, (key, entry, gpu) in zip(value['stages'], STAGES):
        assert stage['stage'] == key and stage['entrypoint'] == entry and stage['GPU'] is gpu
        seconds = stage['maximum_seconds']
        assert isinstance(seconds, (int, float)) and not isinstance(seconds, bool)
        assert math.isfinite(seconds) and 0 < seconds <= STAGE_SECONDS[key]
        assert isinstance(stage['args'], list) and all(isinstance(a, str) for a in stage['args'])
        child_registration = stage['registration']
        assert len(child_registration['sha256']) == 64
        assert Path(child_registration['path']).is_absolute() and '..' not in Path(child_registration['path']).parts
        name = stage['job_name']
        assert name and Path(name).name == name and '/' not in name and '\\' not in name and name not in names
        names.add(name)
        completion = stage['completion']
        path = Path(completion['path'])
        assert path.is_absolute() and '..' not in path.parts
        assert any(path.is_relative_to(base) and path != base for base in roots)
        assert completion['status'] and path not in completions
        completions.add(path)
        output = path.parent
        assert output not in roots, 'Each stage writes only its distinct new output subdirectory'
        expected_args = (['--config', child_registration['path'], '--output', str(output)]
            if key == 'layersCPU' else ['--registration', child_registration['path']])
        assert stage['args'] == expected_args, 'Exact SHA-bound child config/output arguments only'
        if check_files:
            child_config = read_ref(child_registration)
            if key != 'layersCPU':
                assert Path(child_config['output_directory']) == output
    upstream = value['upstream']
    assert [item['arm'] for item in upstream['recipes']] == ARMS
    old_sources = {}
    for item in upstream['recipes']:
        if enforce_paths:
            assert Path(item['path']) == ROOT / 'configs/functional_full_lifecycle_20261022_v1' / (item['arm'] + '.json')
        if check_files:
            recipe = read_ref(item)
            assert recipe['arm'] == item['arm'] and core_sha(recipe) == item['recipe_core_sha256']
            for name, digest in recipe['source_sha256'].items():
                assert Path(name).name == name
                assert name not in old_sources or old_sources[name] == digest
                old_sources[name] = digest
    if check_files:
        for name, digest in old_sources.items():
            assert sha(root / name, SOURCE_CAP) == digest, ('Frozen campaign source changed', name)
        for key in ('wrapper', 'resource_helper', 'authorization'):
            path = value[key]['path']
            assert sha(path, SOURCE_CAP) == value[key]['sha256']
        assert pins['run_gpu_job_v2.py'] == value['wrapper']['sha256']
        assert pins['brainhive_functional_training_contract_v2.py'] == value['resource_helper']['sha256']
        assert sha(upstream['controller']['path'], SOURCE_CAP) == upstream['controller']['sha256']
        auth = read_ref(value['authorization'])
        assert auth['existing_single_GPU_only'] is True and auth['purchase_or_resource_expansion_authorized'] is False
        assert auth['total_GPU_job_hours_cap'] is None
    return value


def terminal_metadata(state, registration, *, reader=read_ref, is_live=live):
    """Inspect closure metadata only; never load score CSVs or feature tensors."""
    status = state.get('status', '')
    if status != SUCCESS:
        if any(token in status for token in ('FAILED', 'FAILURE', 'STOP_REQUESTED', 'STOPPED')):
            raise RuntimeError('Upstream failed or stopped; preserve without diagnostic retry: ' + status)
        return None
    assert not state.get('unresolved_live_child') and state.get('current_stage') is None
    # The upstream writes its success JSON before releasing its flock/exiting.
    # That short, ordinary interval is still a wait, never an analysis failure.
    if is_live(state.get('pid')) or is_live(state.get('child_pid')):
        return None
    completed = state['completed']
    assert [item['arm'] for item in completed] == ARMS
    recipes = {item['arm']: item for item in registration['upstream']['recipes']}
    proofs = []
    for item in completed:
        arm = item['arm']; core = recipes[arm]['recipe_core_sha256']
        for key, expected in (
            ('final', 'FUNCTIONAL_ALIGNMENT_V2_ALL10_REAL_OOF_FIXED_FINAL_COMPONENT_COMPLETE'),
            ('generated', 'FUNCTIONAL_ALIGNMENT_V2_ALL16600_IMAGES_AND_TARGET_JOIN_COMPLETE'),
            ('evaluation', 'FUNCTIONAL_ALIGNMENT_V2_FULL16600_EVALUATION_CLOSED')):
            value = reader(item[key])
            assert value['status'] == expected and value['arm'] == arm and value['recipe_core_sha256'] == core
            if key != 'final':
                assert value['queries'] == 16600
            if key == 'evaluation':
                assert value['actual_full_V16600_evaluation_completed'] is True
                assert value['full_identity_target_seed_noise_chain_verified'] is True
            proofs.append(item[key])
    for attempt in state.get('attempts', []):
        assert attempt.get('exit_code') == 0 or attempt.get('exit_code') is not None
        if 'job' in attempt:
            job = reader(attempt['job'])
            assert job['status'] != 'RUNNING' and not is_live(job.get('pid')) and not is_live(job.get('wrapper_pid'))
    comparison = reader(state['comparison'])
    assert comparison['status'] == COMPARISON
    assert comparison['actual_full_V16600_comparison_completed'] is True
    return dict(status=SUCCESS, completed=completed, comparison=state['comparison'],
        closure_metadata_refs=proofs, score_values_accessed=False)


def prior_children_dead(path):
    """The preserved failed original controller must also be fully stopped."""
    if not Path(path).exists():
        return
    value = read(path)
    assert not value.get('unresolved_live_child')
    assert not live(value.get('pid')) and not live(value.get('child_pid'))
    for attempt in value.get('attempts', []):
        if 'job' in attempt:
            job = read_ref(attempt['job'])
            assert job['status'] != 'RUNNING' and not live(job.get('pid')) and not live(job.get('wrapper_pid'))


def directory_bytes(path):
    total = 0
    path = Path(path)
    if not path.exists():
        return total
    assert not path.is_symlink(), 'Output root cannot be symlink'
    for current, directories, files in os.walk(path):
        for name in directories:
            assert not (Path(current) / name).is_symlink(), 'Output directory cannot be symlink'
        for name in files:
            item = Path(current) / name
            assert not item.is_symlink(), 'Output file cannot be symlink'
            details = item.stat()
            total += max(details.st_size, getattr(details, 'st_blocks', 0) * 512)
    return total


def budget(registration, *, check_disk=True):
    amounts = {Path(path): directory_bytes(path) for path in registration['output_roots']}
    used = sum(amounts.values())
    assert used <= LIMITS['maximum_output_bytes'], 'Independent total analysis output cap exceeded'
    metadata = directory_bytes(BASE)
    assert metadata <= METADATA_CAP, 'Independent analysis metadata cap exceeded'
    if check_disk:
        by_device = {}
        for root, actual in amounts.items():
            metadata_root = root == BASE
            cap = METADATA_CAP if metadata_root else LIMITS['maximum_output_bytes'] - METADATA_CAP
            assert actual <= cap, 'Independent per-volume analysis output cap exceeded'
            extra = LIMITS['minimum_metadata_disk_extra_bytes'] if metadata_root else LIMITS['minimum_disk_extra_bytes']
            path = root
            while not path.exists(): path = path.parent
            volume = by_device.setdefault(path.stat().st_dev, dict(path=path, remaining=0, extra=0))
            volume['remaining'] += cap - actual
            volume['extra'] = max(volume['extra'], extra)
        for volume in by_device.values():
            assert shutil.disk_usage(volume['path']).free >= volume['remaining'] + volume['extra'], 'Insufficient independent per-volume output allocation plus disk buffer'
    return dict(total_output_bytes=used, metadata_bytes=metadata)


def headroom(registration):
    # Imported only after the upstream closure; this helper never imports Torch.
    import brainhive_functional_training_contract_v2 as C
    assert sha(registration['resource_helper']['path'], SOURCE_CAP) == registration['resource_helper']['sha256']
    memory = C.resource_headroom()
    assert memory['effective_headroom_bytes'] >= LIMITS['minimum_CPU_headroom_bytes'], 'Less than10GiB effective CPU headroom'
    return memory


def gpu_free():
    result = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    assert not result.strip(), 'GPU compute process already running'
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    assert len(rows) == 1 and int(rows[0]) * (1 << 20) >= LIMITS['minimum_free_GPU_bytes'], 'Single GPU requires20GiB free'


def stop_owned(process, gpu, job, expected_command, job_name):
    registered = None
    if gpu and not Path(job).exists():
        # The wrapper starts its separate-session child just before writing
        # its receipt. Keep the wrapper alive during this small registration
        # interval; killing it first could orphan the unregistered child.
        deadline = time.monotonic() + 3
        while process.poll() is None and not Path(job).exists() and time.monotonic() < deadline:
            time.sleep(.1)
        if not Path(job).exists():
            return process.poll() is not None
    if gpu and Path(job).exists():
        record = read(job)
        assert record.get('wrapper_pid') == process.pid and record.get('name') == job_name
        assert record.get('command') == expected_command, 'Never kill an unbound GPU child'
        registered = record.get('pid')
        if live(registered):
            try:
                assert os.getpgid(registered) == registered
                os.killpg(registered, signal.SIGTERM)
            except ProcessLookupError:
                pass
    elif process.poll() is None:
        # GPU wrapper has not spawned/recorded a child yet. Terminate only our
        # own wrapper group; never infer a GPU PID from nvidia-smi.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        if registered and live(registered):
            try:
                os.killpg(registered, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif not gpu:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            return False
    return process.poll() is not None and not live(registered)


class Runner:
    def __init__(self, registration, registration_path):
        assert not STATE.exists(), 'Preserve prior queue; no automatic retry or overwrite'
        self.registration = registration
        self.registration_path = Path(registration_path)
        self.state = dict(schema=SCHEMA, status='WAITING_FOR_COMPLETE_FOUR_ARM_LIFECYCLE_AND_COMPARISON',
            pid=os.getpid(), child_pid=None, registration=ref(registration_path),
            source_sha256=sha(__file__, SOURCE_CAP), upstream_state=registration['upstream']['state_path'],
            attempts=[], completed=[], current_stage=None, no_candidate_fit=True,
            auto_fit=False, fitted_parameters=0, score_values_accessed=False, automatic_retry=False)
        self.save()

    def save(self):
        atomic(STATE, self.state)

    def wait_upstream(self):
        import fcntl
        deadline = time.monotonic() + LIMITS['wait_seconds']
        while True:
            assert time.monotonic() < deadline, 'Finite48h upstream waiting boundary reached'
            # All these files are small and immutable; no model/cache hashing.
            assert sha(self.registration_path, JSON_CAP) == self.state['registration']['sha256']
            validate_registration(self.registration)
            upstream = read(self.registration['upstream']['state_path'])
            proof = terminal_metadata(upstream, self.registration)
            if proof is not None:
                prior_children_dead(UPSTREAM_BASE / 'formal_queue.json')
                lock = Path(self.registration['upstream']['lifecycle_lock']).open('a')
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lock.close()
                else:
                    # Close the check/lock race using a second complete check.
                    assert terminal_metadata(read(self.registration['upstream']['state_path']), self.registration) == proof
                    self.state.update(upstream_closure=proof, upstream_terminal_snapshot=ref(self.registration['upstream']['state_path']))
                    self.save()
                    return lock
            self.state.update(waiting_status=upstream.get('status'),
                waiting_completed_arms=[item['arm'] for item in upstream.get('completed', [])],
                wait_elapsed_seconds=LIMITS['wait_seconds'] - max(0, deadline - time.monotonic()))
            self.save()
            time.sleep(LIMITS['poll_seconds'])

    def stage(self, stage):
        assert sha(self.registration_path, JSON_CAP) == self.state['registration']['sha256']
        validate_registration(self.registration)
        self.state.update(status='RUNNING_FIXED_FINITE_T_ONLY_ANALYSIS', current_stage=stage['stage'],
            current_resource_headroom=headroom(self.registration), current_budget=budget(self.registration))
        self.save()
        if stage['GPU']:
            gpu_free()
        entry = ROOT / stage['entrypoint']
        child = [sys.executable, '-B', '-u', str(entry), *stage['args']]
        name = stage['job_name']; seconds = stage['maximum_seconds']
        job = ROOT / 'audit' / (name + '_job.json')
        if stage['GPU']:
            assert not job.exists(), 'Preserve previous GPU receipt; no retry'
            argv = [sys.executable, '-B', str(ROOT / 'run_gpu_job_v2.py'), '--name', name,
                '--max-seconds', str(seconds), '--', *child]
        else:
            argv = child
        log = BASE / (name + '.log')
        assert not log.exists() and not Path(stage['completion']['path']).exists(), 'Fresh bounded stage only'
        env = os.environ.copy()
        if not stage['GPU']:
            env['CUDA_VISIBLE_DEVICES'] = ''
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[key] = '4' if stage['stage'] == 'layersCPU' else '2'
        attempt = dict(stage=stage['stage'], command=argv, child_command=child,
            maximum_seconds=seconds, GPU=stage['GPU'], log=str(log),
            registration=self.state['registration'], stage_registration=stage['registration'], entrypoint=ref(entry))
        self.state['attempts'].append(attempt); self.save()
        started = time.monotonic()
        with log.open('xb') as stream:
            process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True, shell=False)
            attempt['pid'] = process.pid
            self.state['child_pid'] = process.pid; self.save()
            try:
                while process.poll() is None:
                    assert log.stat().st_size <= LOG_CAP, 'Finite16MiB stage log cap'
                    budget(self.registration)
                    headroom(self.registration)
                    assert time.monotonic() - started <= seconds + (120 if stage['GPU'] else 0), 'Fixed stage timeout'
                    time.sleep(2)
            except BaseException as error:
                attempt['interruption'] = repr(error)
                try:
                    terminal = stop_owned(process, stage['GPU'], job, child, name)
                except BaseException as stop_error:
                    terminal = False; attempt['stop_error'] = repr(stop_error)
                self.state.update(unresolved_live_child=not terminal,
                    child_pid=None if terminal else process.pid)
                attempt.update(exit_code=process.poll(), seconds=time.monotonic() - started,
                    log_sha256=sha(log), log_bytes=log.stat().st_size)
                self.save()
                raise
        attempt.update(exit_code=process.returncode, seconds=time.monotonic() - started,
            log_sha256=sha(log), log_bytes=log.stat().st_size)
        self.state['child_pid'] = None; self.save()
        assert log.stat().st_size <= LOG_CAP
        if stage['GPU']:
            record = read(job)
            assert record['wrapper_pid'] == process.pid and record['command'] == child and record['name'] == name
            assert not live(record.get('pid')) and not live(record.get('wrapper_pid'))
            attempt['job'] = ref(job)
            assert record['status'] == 'COMPLETED' and record['exit_code'] == 0
            assert record['authorization_sha256'] == self.registration['authorization']['sha256']
        assert process.returncode == 0, 'Stage failure retained; no automatic retry'
        completion = read(stage['completion']['path'])
        assert completion['status'] == stage['completion']['status'], 'Expected complete bounded stage evidence'
        assert completion.get('auto_fit', False) is False
        assert completion.get('fitted_parameters', 0) == 0
        result = dict(stage=stage['stage'], completion=ref(stage['completion']['path']),
            log=dict(path=str(log), sha256=attempt['log_sha256'], bytes=attempt['log_bytes']),
            registration=self.state['registration'], stage_registration=stage['registration'])
        for key in ('input_sha256', 'input_provenance', 'output_refs', 'source_sha256'):
            if key in completion:
                result[key] = completion[key]
        self.state['completed'].append(result)
        self.state['current_budget'] = budget(self.registration)
        self.save()


def preflight(registration_path):
    assert os.name == 'posix' and ROOT == SERVER, 'Run only in the registered server repository'
    value = validate_registration(read(registration_path))
    # Only metadata/source validation, no GPU query, Torch, large cache hash,
    # or requirement that the still-running upstream has already completed.
    upstream = read(value['upstream']['state_path'])
    return value, dict(status='PREFLIGHT_ONLY_FIXED_ANALYSIS_CONTRACT_VALID',
        upstream_status=upstream.get('status'), stages=[item['stage'] for item in value['stages']],
        no_GPU_access=True, no_Torch_import=True, no_large_input_hash=True,
        score_values_accessed=False, queue_started=False, registration=ref(registration_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', required=True, type=Path)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    registration, report = preflight(args.registration)
    if args.preflight:
        print(json.dumps(report)); return
    import fcntl
    BASE.mkdir(parents=True, exist_ok=True)
    with (BASE / 'analysis_controller.lock').open('a') as own_lock:
        fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner = Runner(registration, args.registration)
        upstream_lock = None
        def requested_stop(signum, frame):
            raise InterruptedError('Analysis controller received signal ' + str(signum))
        signal.signal(signal.SIGTERM, requested_stop)
        try:
            upstream_lock = runner.wait_upstream()
            for stage in registration['stages']:
                runner.stage(stage)
            runner.state.update(status='ALL_FIXED_T_ONLY_DECISIVE_ANALYSES_COMPLETE_NO_FIT',
                current_stage=None, child_pid=None, auto_fit=False,
                new_method_proven=False, independent_SOTA_established=False)
            runner.save()
        except BaseException as error:
            runner.state.update(status=('ANALYSIS_STOP_REQUESTED_OWN_CHILD_UNRESOLVED'
                if runner.state.get('unresolved_live_child') else 'ANALYSIS_FAILED_PRESERVED_NO_AUTOMATIC_RETRY'),
                error=repr(error))
            runner.save(); raise
        finally:
            if upstream_lock is not None:
                upstream_lock.close()


if __name__ == '__main__':
    main()
