"""Finite real CPU delivery of unadmitted full-train input/profile programs."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'audit/official_input_pipeline_delivery_20261022_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def headroom():
    physical = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                    if line.startswith('MemAvailable:'))
    effective = min(physical, int(Path('/sys/fs/cgroup/memory.max').read_text())
                    - int(Path('/sys/fs/cgroup/memory.current').read_text()))
    assert effective >= 12 << 30, 'Preserve current formal fit8GiB floor and bounded CPU-check headroom'
    return effective


def actual_current_training():
    processes = []
    for pid, name in [(4150, 'queue_BrainHIVE_functional_recovery_20261022_v1.py'),
                      (4180, 'train_BrainHIVE_functional_alignment_v2.py')]:
        path = Path('/proc') / str(pid)
        command = path.joinpath('cmdline').read_bytes().replace(b'\0', b' ').decode()
        state = path.joinpath('stat').read_text().rsplit(') ', 1)[1].split()[0]
        assert name in command and state not in ('Z', 'X'), (pid, state, command)
        processes.append(dict(pid=pid, entry=name, live=True))
    return processes


def main():
    assert os.name == 'posix'
    folder = OUTPUT / 'server_CPU_attempt1'
    assert not folder.exists(), 'Retain every earlier actual CPU attempt'
    pins = json.loads((OUTPUT / 'implementation_sources.json').read_bytes())
    original = json.loads((ROOT / 'configs/BrainHIVE_new_instance_continuation_sources_20261022_v2.json').read_bytes())['source_sha256']
    previous = json.loads((ROOT / 'audit/official_executor_implementation_delivery_20261022_v1/actual_progress.json').read_bytes())

    def source_check():
        assert len(original) == 29 and len(pins['sources']) == 6
        for name, expected in {**original, **previous['implementation_sources'], **pins['sources']}.items():
            assert sha(ROOT / name) == expected, name
        for value in previous['unchanged_current_recipes'].values(): assert sha(value['path']) == value['sha256']
        assert sha(ROOT / 'results.tsv') == previous['results_table']['sha256']

    source_check(); memory_before = headroom(); before = actual_current_training()
    official_root = Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/official_fullfit_v1')
    assert not official_root.exists(), 'CPU delivery must not create/adopt formal official outputs'
    folder.mkdir(mode=0o700)
    running = dict(status='ACTUAL_BOUNDED_CPU_INPUT_PIPELINE_CHECKS_RUNNING', pid=os.getpid(),
        UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), source_sha256=sha(__file__),
        implementation_source_manifest=dict(path=str(OUTPUT / 'implementation_sources.json'), sha256=sha(OUTPUT / 'implementation_sources.json')))
    (folder / 'process.json').write_bytes((json.dumps(running, indent=2) + '\n').encode())
    record = dict(running, stages=[], actual_training_before=before, effective_CPU_headroom_before_bytes=memory_before,
        actual_GPU_jobs=0, actual_formal_fits=0, actual_official_condition_exports=0,
        actual_full16540_latent_banks=0, real_EEG_image_or_weight_payloads_read=False,
        official_or_new_V_metric_values_read=False, old_files_deleted=0, scientific_goal_complete=False)
    plan = str(ROOT / 'audit/official_fullfit_metadata_plan_server_20261022_v1')
    cases = [
        ('component_loader', [sys.executable, '-B', 'check_BrainHIVE_official_component_data_CPU_v1.py', '--plan-directory', plan, '-v'], 10),
        ('latent_bank', [sys.executable, '-B', 'check_BrainHIVE_official_train_latents_CPU_v1.py', '--receipt', str(folder / 'latent_checks.json')], 12),
        ('encoder_profile', [sys.executable, '-B', 'check_BrainHIVE_official_encoder_profile_CPU_v1.py', '--receipt', str(folder / 'profile_checks.json')], 9),
    ]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    started = time.monotonic()
    try:
        for name, command, expected in cases:
            pre = headroom(); step = time.monotonic()
            completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, timeout=240)
            raw = completed.stdout + completed.stderr
            assert len(raw) <= 512 << 10
            log = folder / (name + '.log'); log.write_bytes(raw)
            record['stages'].append(dict(name=name, command=command, expected_checks=expected, exit_code=completed.returncode,
                seconds=time.monotonic() - step, CPU_headroom_before_bytes=pre, CPU_headroom_after_bytes=headroom(),
                log=dict(path=str(log), sha256=sha(log), bytes=len(raw))))
            assert completed.returncode == 0, raw.decode(errors='replace')[-6000:]
            if name == 'component_loader':
                assert re.search(rb'Ran 10 tests', raw) and b'\nOK' in raw
            elif name == 'latent_bank':
                proof = json.loads((folder / 'latent_checks.json').read_bytes())
                assert proof['checks'] == 12 and proof['failures'] == proof['errors'] == 0 and proof['actual_full16540_bank_built'] is False
            else:
                proof = json.loads((folder / 'profile_checks.json').read_bytes())
                assert len(proof['checks']) == 9 and proof['status'] == 'PASS_TINY_CPU_ENGINEERING_ONLY_NO_PERMISSION_OR_GPU'
                assert proof['actual_GPU_used'] is False and proof['actual_engineering_permission_created'] is False
        source_check(); record['actual_training_after'] = actual_current_training()
        assert not official_root.exists()
        record.update(status='OFFICIAL_FULL_INPUT_AND_PROFILE_31_CPU_CHECKS_PASSED_NO_EXECUTION_ADMISSION',
            actual_checks=31, all29_current_sources_and6_previous_executor_sources_unchanged=True,
            four_current_recipes_and_results_unchanged=True, actual_full165400_tensor_assembly_performed=False,
            actual_full165400_metadata_mapping_verified=True, actual_full16540_image_metadata_verified=True,
            actual_CUDA_or_VAE_encoding_performed=False, official_output_namespace_created=False,
            official_method_lock_or_stage_admission_issued=False)
    except BaseException as error:
        record.update(status='ACTUAL_CPU_INPUT_PIPELINE_CHECK_PARTIAL_FAILURE_RETAINED',
            error_type=type(error).__name__, error=str(error)[:6000])
        raise
    finally:
        record['seconds'] = time.monotonic() - started
        (folder / 'completion.json').write_bytes((json.dumps(record, indent=2, allow_nan=False) + '\n').encode())
    print(json.dumps(dict(status=record['status'], actual_checks=record['actual_checks'], receipt=str(folder / 'completion.json'))))


if __name__ == '__main__': main()
