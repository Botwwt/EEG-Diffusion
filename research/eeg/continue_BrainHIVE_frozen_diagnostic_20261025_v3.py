"""One explicit GPU diagnostic recovery followed by the unchanged fixed readout.

The completed CPU diagnostics are bound and reused. No fit, candidate search,
score tuning, automatic retry, source replacement, or previous-output overwrite.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import queue_BrainHIVE_decisive_T_analysis_20261025_v2 as Q

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'audit/diagnostic_memory_recovery_20261025_v3'
DATA = Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/analysis/diagnostic_memory_recovery_20261025_v3')


def validate(config):
    assert config['schema'] == 'BrainHIVE_frozen_diagnostic_explicit_continuation_20261025_v3'
    assert config['automatic_retry'] is False and config['automatic_new_fit'] is False
    for name, digest in config['source_sha256'].items():
        assert Path(name).name == name and Q.sha(ROOT / name) == digest, name
    recovery = Q.read_ref(config['recovery_registration'])
    profile = Q.read_ref(recovery['profile_success'])
    assert profile['status'] == 'EXACT_LOW_MEMORY_NATIVE_LOAD_PROFILE_PASSED'
    assert profile['peak_extra_CPU_bytes'] <= 10 << 30 and profile['peak_GPU_bytes'] <= 20 << 30
    assert recovery['source_sha256']['run_BrainHIVE_frozen_function_low_memory_20261025_v3.py'] == config['source_sha256']['run_BrainHIVE_frozen_function_low_memory_20261025_v3.py']
    function = Q.read_ref(recovery['diagnostic_registration'])
    assert function['output_directory'] == str(DATA / 'function')
    assert function['gpu_job_name'] == 'BrainHIVE_decisive_T_function_20261025_v3'
    assert function['maximum_extra_CPU_bytes'] == 10 << 30
    assert function['maximum_GPU_bytes'] == 20 << 30 and function['maximum_seconds'] == 7200
    for item in config['completed_CPU_diagnostics'].values():
        Q.read_ref(item)
    direction = Q.read_ref(config['direction_registration'])
    assert direction['output_directory'] == str(DATA / 'direction')
    assert direction['summaries']['function']['path'] == str(DATA / 'function/function_summary.json')
    upstream = Q.read_ref(config['upstream_closed_queue'])
    assert upstream['status'] == Q.SUCCESS
    assert not Q.live(upstream['pid']) and not Q.live(upstream.get('child_pid'))
    assert [row['arm'] for row in upstream['completed']] == Q.ARMS
    assert Q.read_ref(upstream['comparison'])['status'] == Q.COMPARISON
    previous = Q.read_ref(config['failed_analysis_queue'])
    assert previous['status'] == 'ANALYSIS_FAILED_PRESERVED_NO_AUTOMATIC_RETRY'
    assert not Q.live(previous['pid']) and not Q.live(previous.get('child_pid'))
    assert not previous.get('unresolved_live_child')
    import brainhive_functional_training_contract_v2 as C
    memory = C.resource_headroom()
    assert memory['effective_headroom_bytes'] >= 10 << 30
    assert shutil.disk_usage(DATA).free >= (1 << 30) + (5 << 30)
    assert shutil.disk_usage(BASE).free >= (64 << 20) + (1 << 30)
    return recovery, function, memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', required=True, type=Path)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    config_sha = Q.sha(args.registration)
    config = Q.read(args.registration)
    recovery, function, memory = validate(config)
    if args.preflight_only:
        print(json.dumps(dict(status='EXPLICIT_CONTINUATION_CLOSED_INPUTS_CPU_PREFLIGHT_PASSED',
            no_GPU_job_started=True, memory=memory)))
        return
    import fcntl
    BASE.mkdir(exist_ok=True)
    state_path = BASE / 'continuation_state.json'
    assert not state_path.exists(), 'Retain prior explicit continuation; no automatic retry'
    with (ROOT / 'audit/functional_full_lifecycle_20261022_v1/lifecycle_controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(status='RUNNING_EXPLICIT_FROZEN_GPU_DIAGNOSTIC_RECOVERY', pid=os.getpid(),
            registration=Q.ref(args.registration), child_pid=None, attempts=[], completed=[],
            automatic_retry=False, automatic_new_fit=False, parameter_updates=0)
        def save(): Q.atomic(state_path, state)
        def interrupted(signum, frame): raise InterruptedError('Continuation received signal ' + str(signum))
        signal.signal(signal.SIGTERM, interrupted)
        save()
        stages = [
            ('functionGPU', 'BrainHIVE_decisive_T_function_20261025_v3', 7200, True,
             [sys.executable, '-B', '-u', str(ROOT / 'run_BrainHIVE_frozen_function_low_memory_20261025_v3.py'),
              '--registration', config['recovery_registration']['path']], DATA / 'function/completion.json',
             'COMPLETE_FROZEN_T_ONLY_FUNCTIONAL_DIAGNOSTIC'),
            ('directionCPU', 'BrainHIVE_fixed_T_direction_20261025_v3', 300, False,
             [sys.executable, '-B', str(ROOT / 'read_BrainHIVE_decisive_T_analysis_CPU_20261025_v2.py'),
              '--registration', config['direction_registration']['path']], DATA / 'direction/direction_summary.json',
             'COMPLETE_FIXED_T_ONLY_DIRECTION_READOUT')]
        try:
            for stage, name, seconds, gpu, child, completion, expected in stages:
                assert Q.sha(args.registration) == config_sha
                validate(config)
                if gpu: Q.gpu_free()
                job = ROOT / 'audit' / (name + '_job.json')
                assert not job.exists() and not completion.exists()
                argv = ([sys.executable, '-B', str(ROOT / 'run_gpu_job_v2.py'), '--name', name,
                         '--max-seconds', str(seconds), '--', *child] if gpu else child)
                log = BASE / (name + '.log')
                assert not log.exists()
                attempt = dict(stage=stage, command=argv, child_command=child, log=str(log), maximum_seconds=seconds)
                state['attempts'].append(attempt); state['current_stage'] = stage; save()
                process = None
                started = time.monotonic()
                env = os.environ.copy()
                for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
                    env[key] = '2'
                if not gpu: env['CUDA_VISIBLE_DEVICES'] = ''
                with log.open('xb') as stream:
                    try:
                        process = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                        state['child_pid'] = attempt['pid'] = process.pid; save()
                        while process.poll() is None:
                            assert time.monotonic() - started <= seconds + (120 if gpu else 0)
                            assert log.stat().st_size <= 16 << 20
                            assert Q.directory_bytes(DATA) <= 1 << 30
                            assert Q.directory_bytes(BASE) <= 64 << 20
                            import brainhive_functional_training_contract_v2 as C
                            assert C.resource_headroom()['effective_headroom_bytes'] >= 10 << 30
                            time.sleep(2)
                        if gpu:
                            record = Q.read(job)
                            assert record['wrapper_pid'] == process.pid and record['command'] == child
                            assert not Q.live(record['pid']) and not Q.live(record['wrapper_pid'])
                            assert record['status'] == 'COMPLETED' and record['exit_code'] == 0
                            attempt['job'] = Q.ref(job)
                    except BaseException:
                        if process is not None:
                            attempt['owned_cleanup'] = {}
                            terminal = Q.stop_owned(process, gpu, job, child, name, attempt['owned_cleanup'])
                            state['unresolved_live_child'] = not terminal
                            state['child_pid'] = None if terminal else process.pid
                        save()
                        raise
                attempt.update(exit_code=process.returncode, seconds=time.monotonic() - started, log_sha256=Q.sha(log))
                state['child_pid'] = None; save()
                assert process.returncode == 0
                assert Q.read(completion)['status'] == expected
                state['completed'].append(dict(stage=stage, completion=Q.ref(completion))); save()
            state.update(status='UNCHANGED_FROZEN_FUNCTION_AND_FIXED_DIRECTION_READOUT_COMPLETE', current_stage=None)
            save()
        except BaseException as error:
            state.update(status='EXPLICIT_RECOVERY_FAILED_PRESERVED_NO_AUTOMATIC_RETRY', error=repr(error))
            save()
            raise


if __name__ == '__main__':
    main()
