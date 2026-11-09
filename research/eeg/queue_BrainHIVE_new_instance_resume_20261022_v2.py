"""Finite, evidence-gated continuation after the user-provided instance change.

Restoration, evaluator probe, exact-core profiles, admission, and formal work
remain distinct stages. No cleanup, performance pilot, score-based branch, or
alternative seed. Failed/partial stages retain all evidence and stop the queue.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'audit/new_instance_restore_20261022_v2'
OUT = BASE / 'continuation_queue.json'
AUTH = ROOT / 'configs/resource_migration_authorization_20261022_v4.json'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def live(pid):
    path = Path('/proc') / str(pid) / 'cmdline'
    try: return bool(path.read_bytes())
    except (FileNotFoundError, ProcessLookupError): return False


def save(state):
    state['observation_UTC'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    temporary = OUT.with_name(OUT.name + '.pending')
    with temporary.open('w') as stream:
        json.dump(state, stream, indent=2, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(OUT)


def wait_restoration(state, latent_job, public_pid, evaluation_pid):
    deadline = time.monotonic() + 6000
    while True:
        assert time.monotonic() < deadline, 'Finite runtime restoration wait expired'
        try:
            job = read(latent_job)
        except (json.JSONDecodeError, FileNotFoundError):
            assert live(state['latent_wrapper_pid']), 'Latent wrapper disappeared without a readable terminal receipt'
            state.update(stage='WAITING_SAME_LIVE_LATENT_WRAPPER_DURING_METADATA_WRITE')
            save(state); time.sleep(.5); continue
        if job['status'] != 'RUNNING':
            assert job['status'] == 'COMPLETED' and job['exit_code'] == 0, 'Latent restoration failed'
        files = [BASE / (n + '_driver.json') for n in ('public', 'inputs', 'conditions', 'latent', 'evaluation')]
        if (all(p.is_file() for p in files) and job['status'] == 'COMPLETED'
                and not live(job['pid']) and not live(state['latent_wrapper_pid'])):
            break
        assert live(public_pid) or (BASE / 'public_driver.json').exists(), 'Public producer stopped without durable closure'
        assert live(evaluation_pid) or (BASE / 'evaluation_driver.json').exists(), 'Evaluator producer stopped without durable closure'
        state.update(stage='WAITING_FOR_ACTUAL_RUNTIME_RESTORATION_AND_DURABLE_COPIES',
            public_producer_live=live(public_pid), evaluation_producer_live=live(evaluation_pid),
            latent_job_status=job['status'], completed_driver_receipts=[str(p) for p in files if p.exists()])
        save(state); time.sleep(10)


def run(state, label, entry, arguments, seconds, *, GPU_name=None):
    assert entry in state['source_sha256'] and sha(ROOT / entry) == state['source_sha256'][entry]
    log = BASE / ('continuation_' + label + '.log'); assert not log.exists()
    child = [sys.executable, '-B', '-u', str(ROOT / entry), *arguments]
    job_path = ROOT / 'audit' / (GPU_name + '_job.json') if GPU_name else None
    command = ([sys.executable, '-B', str(ROOT / 'run_gpu_job_v2.py'), '--name', GPU_name,
        '--max-seconds', str(seconds), '--', *child] if GPU_name else child)
    if job_path: assert not job_path.exists()
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '2'
    attempt = dict(stage=label, argv=command, max_seconds=seconds, log=str(log))
    state['attempts'].append(attempt); state.update(stage=label); save(state)
    started = time.monotonic()
    with log.open('xb') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
            stderr=subprocess.STDOUT, shell=False, start_new_session=True)
        state['child_pid'] = process.pid; save(state)
        try:
            while process.poll() is None:
                assert log.stat().st_size < 32 << 20, 'Finite continuation log budget'
                assert time.monotonic() - started < seconds + (60 if GPU_name else 0), 'Finite stage duration'
                if label == 'formal_full_four_arm_lifecycle':
                    nested = ROOT / 'audit/functional_full_lifecycle_20261022_v1/formal_queue.json'
                    if nested.exists():
                        current = read(nested)
                        state['formal_controller_status'] = current['status']
                        state['formal_controller_current_stage'] = current.get('current_stage')
                        for arm in ('O', 'P', 'PN', 'S'):
                            config = ROOT / 'configs/functional_full_lifecycle_20261022_v1' / (arm + '.json')
                            if not config.exists(): continue
                            for name in read(config)['formal_job_names']:
                                actual = ROOT / 'audit' / (name + '_job.json')
                                if actual.exists():
                                    try: proof = read(actual)
                                    except (FileNotFoundError, json.JSONDecodeError): continue
                                    if (proof['status'] == 'RUNNING' and live(proof['pid'])) or proof['status'] == 'COMPLETED':
                                        state['formal_training_started'] = True
                                        state['actual_formal_training_job'] = dict(path=str(actual), status=proof['status'], pid=proof['pid'])
                        save(state)
                time.sleep(5)
        except BaseException:
            if label in ('new_core_four_GPU_profiles', 'formal_full_four_arm_lifecycle') and process.poll() is None:
                state.update(unresolved_live_child=process.pid,
                    status='EXPLICIT_OBSERVATION_FAILURE_FINITE_INNER_CONTROLLER_STILL_LIVE')
                save(state)
                raise
            if GPU_name:
                try: job = read(job_path)
                except (FileNotFoundError, json.JSONDecodeError): job = {}
                if job.get('wrapper_pid') == process.pid and live(job.get('pid')):
                    try: os.killpg(job['pid'], signal.SIGTERM)
                    except ProcessLookupError: pass
                # Do not treat an observation timeout as terminal. Wait this exact
                # wrapper so that it can close the registered compute receipt.
                try: process.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    state.update(unresolved_live_child=process.pid, status='EXPLICIT_STOP_WITH_UNRESOLVED_LIVE_WRAPPER')
                    save(state)
                    raise
            elif process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=15)
            raise
    attempt.update(exit_code=process.returncode, seconds=time.monotonic()-started, log_sha256=sha(log))
    state['child_pid'] = None; save(state)
    assert process.returncode == 0, 'Stage failed; preserve exact log and stop'
    if job_path:
        job = read(job_path)
        assert job['status'] == 'COMPLETED' and job['exit_code'] == 0 and not live(job['pid'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-manifest', required=True, type=Path)
    parser.add_argument('--public-pid', required=True, type=int)
    parser.add_argument('--evaluation-pid', required=True, type=int)
    parser.add_argument('--latent-wrapper-pid', required=True, type=int)
    args = parser.parse_args()
    assert os.name == 'posix' and ROOT == Path('/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1')
    assert not OUT.exists()
    manifest = read(args.source_manifest)
    assert manifest['status'] == 'SOURCE_BOUND_REPLACEMENT_INSTANCE_CONTINUATION_READY'
    for name, digest in manifest['source_sha256'].items(): assert sha(ROOT / name) == digest
    authorization = read(AUTH)
    assert authorization['cleanup_cancelled'] is True and authorization['training_seed'] == 1729
    assert authorization['use_replacement_existing_single_GPU'] is True
    assert not authorization['formal_training_admission_implied_by_this_record']
    latent_job = ROOT / 'audit/BrainHIVE_new_instance_T_latents_20261022_v2_job.json'
    state = dict(status='RUNNING_EVIDENCE_GATED_REPLACEMENT_INSTANCE_CONTINUATION', pid=os.getpid(),
        start_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), source_sha256=manifest['source_sha256'],
        authorization_sha256=sha(AUTH), source_manifest_sha256=sha(args.source_manifest),
        stage=None, attempts=[], child_pid=None, cleanup_performed=False,
        latent_wrapper_pid=args.latent_wrapper_pid,
        formal_training_started=False, new_scientific_result_claimed=False, goal_complete=False)
    save(state)
    try:
        wait_restoration(state, latent_job, args.public_pid, args.evaluation_pid)
        run(state, 'exact_runtime_closure', 'verify_BrainHIVE_new_instance_runtime_closure_CPU_20261022_v2.py',
            ['--evaluation', '--attempt', '1'], 600)
        closure = read(BASE / 'exact_coverage_closure_core_and_evaluation_attempt1.json')
        assert closure['status'] == 'EXACT_ORIGINAL_IDENTITY_COVERAGE_RAM_AND_DURABLE_RUNTIME_CLOSURE_VERIFIED'
        assert closure['core_original_records'] == 55 and closure['evaluator_original_records'] == 25
        assert closure['direct_profile_input_SHA_count'] == 37
        run(state, 'evaluation_GPU_probe', 'probe_BrainHIVE_evaluation_environment_GPU_20261022_v2.py',
            ['--asset-receipt', str(ROOT / 'audit/runtime_evaluation_public_restore_20261022_v1_attempt1.json'), '--attempt', '1'],
            900, GPU_name='BrainHIVE_evaluation_environment_probe_20261022_v2_attempt1')
        for name, digest in manifest['source_sha256'].items(): assert sha(ROOT / name) == digest
        run(state, 'prepare_complete_recipes', 'prepare_BrainHIVE_functional_lifecycle_CPU_20261022_v1.py', ['prepare'], 600)
        run(state, 'new_core_four_GPU_profiles', 'queue_BrainHIVE_functional_lifecycle_20261022_v1.py', ['--mode', 'profiles'], 7500)
        readiness = ROOT / 'audit/new_instance_evaluation_readiness_20261022_v2_attempt1.json'
        run(state, 'formal_admission', 'prepare_BrainHIVE_functional_lifecycle_CPU_20261022_v1.py',
            ['admit', '--evaluation-readiness', str(readiness)], 600)
        state.update(stage='RUNNING_ADMITTED_FORMAL_LIFECYCLE_CONTROLLER'); save(state)
        run(state, 'formal_full_four_arm_lifecycle', 'queue_BrainHIVE_functional_lifecycle_20261022_v1.py',
            ['--mode', 'formal'], 6 * 86400 + 300)
        state.update(status='FULL_FOUR_ARM_LIFECYCLE_CONTROLLER_COMPLETED_SCIENTIFIC_READOUT_PENDING', stage=None)
        save(state)
    except BaseException as error:
        if not state.get('unresolved_live_child'):
            state.update(status='CONTINUATION_STOPPED_WITH_PRESERVED_EXPLICIT_FAILURE_NO_AUTOMATIC_RETRY')
        state.update(error_type=type(error).__name__, error=str(error)[:900])
        save(state)
        raise


if __name__ == '__main__':
    main()
