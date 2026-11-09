"""Finite real CPU delivery checks for unadmitted official executors."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'audit/official_executor_implementation_delivery_20261022_v1'
PINS = {
    'brainhive_official_encoder_training_v1.py': 'ddcab9592b7a495c33ba66bdcba4134b75dc00774de0157ef044ac588ef74451',
    'train_BrainHIVE_official_encoder_v1.py': '8666f4ced0e0c3d70b5453601d16992e543721de060992b442bd9cb1e18dcedc',
    'check_BrainHIVE_official_encoder_training_CPU_v1.py': 'a5c0268294e63b2c28bdf502090b05e20297d3e47522cc822ea2e004abf5e8ed',
    'brainhive_official_execution_contract_CPU_v1.py': '13ba7c9831d6bac8fe7b208a615e1a36758ba9c52727faa1755648f63321ac0c',
    'export_BrainHIVE_official_OOF_conditions_v1.py': 'cf5db8ee88412227c946240ea52d6874722e825b4f307a9b71b3a4618319685d',
    'check_BrainHIVE_official_execution_and_export_CPU_v1.py': 'c23c07a6a9f144a02b4de6f1280741bcabfa536ba2395d90a172a258f6294fc7',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def headroom():
    available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    bound = int(Path('/sys/fs/cgroup/memory.max').read_text()) - int(Path('/sys/fs/cgroup/memory.current').read_text())
    value = min(available, bound)
    assert value >= 12 << 30, 'Leave ongoing formal training its fixed CPU protection'
    return value


def main():
    assert os.name == 'posix'
    completion = OUTPUT / 'server_execution.json'
    assert not completion.exists(), 'Retain previous actual evidence; no automatic retry'
    assert not (ROOT / 'audit/official_fullfit_execution_review_v1.json').exists(), 'This delivery only checks missing-review refusal'
    frozen = json.loads((ROOT / 'configs/BrainHIVE_new_instance_continuation_sources_20261022_v2.json').read_text())['source_sha256']
    assert len(frozen) == 29
    for name, digest in dict(frozen, **PINS).items(): assert sha(ROOT / name) == digest
    output_namespace = Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/official_fullfit_v1')
    assert not output_namespace.exists(), 'No official task output may be adopted by a CPU checker'
    OUTPUT.mkdir(exist_ok=True)
    started = time.monotonic()
    record = dict(status='UNADMITTED_OFFICIAL_EXECUTOR_CPU_CHECKS_RUNNING', UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        source_sha256=sha(__file__), implementation_source_sha256=PINS, stages=[],
        method_lock_or_execution_admission_issued=False, official_fit_updates=0, actual_official_condition_exports=0,
        actual_dataset_or_pretrained_payloads_read=False, actual_GPU_jobs=0, official_or_new_V_metric_values_read=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    plan = str(ROOT / 'audit/official_fullfit_metadata_plan_server_20261022_v1')
    stages = [
        ('training_checks', [sys.executable, '-B', 'check_BrainHIVE_official_encoder_training_CPU_v1.py'], 0, 90),
        ('execution_export_checks', [sys.executable, '-B', 'check_BrainHIVE_official_execution_and_export_CPU_v1.py',
            '--plan-directory', plan, '--receipt', str(OUTPUT / 'execution_export_checks.json')], 0, 90),
    ]
    for name in ('train_BrainHIVE_official_encoder_v1.py', 'export_BrainHIVE_official_OOF_conditions_v1.py'):
        stages.append((name[:-3] + '_actual_missing_review', [sys.executable, '-B', name,
            '--plan-directory', plan, '--task-id', 'sub-01/official_oof0'], 1, 20))
    try:
        for name, command, expected, maximum in stages:
            before = headroom(); step = time.monotonic()
            completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, timeout=maximum)
            log = OUTPUT / (name + '.log'); assert not log.exists()
            raw = completed.stdout + completed.stderr; assert len(raw) <= 128 << 10
            log.write_bytes(raw)
            record['stages'].append(dict(name=name, command=command, expected_exit_code=expected,
                exit_code=completed.returncode, seconds=time.monotonic() - step,
                CPU_headroom_before_bytes=before, CPU_headroom_after_bytes=headroom(),
                log=dict(path=str(log), sha256=sha(log), bytes=len(raw))))
            assert completed.returncode == expected, raw.decode(errors='replace')
            if expected == 1:
                assert b'Official method/resource execution review does not exist' in raw
        assert not output_namespace.exists() and not (ROOT / 'audit/official_fullfit_execution_review_v1.json').exists()
        for name, digest in dict(frozen, **PINS).items(): assert sha(ROOT / name) == digest
        assert sha(ROOT / 'results.tsv') == '2ee905cc31949671c3068f04c9dbd05046ed4ae9888b05c9ca7cb08d781fb010'
        record.update(status='OFFICIAL_EXECUTOR_IMPLEMENTATION_CPU_CHECKS_AND_REAL_EARLY_REFUSALS_VERIFIED_NO_ADMISSION',
            training_CPU_checks=8, execution_export_CPU_groups=10, actual_entry_missing_review_refusals=2,
            unchanged_frozen_sources=29, official_output_namespace_created=False,
            actual_CUDA_BF16_or_full_dataset_fitting_verified=False,
            current_four_arm_jobs_changed=False, scientific_goal_complete=False)
    except BaseException as error:
        record.update(status='OFFICIAL_EXECUTOR_CPU_CHECKS_FAILED_PARTIAL_EVIDENCE_RETAINED', error_type=type(error).__name__, error=str(error)[:1200])
        raise
    finally:
        record['seconds'] = time.monotonic() - started
        completion.write_bytes((json.dumps(record, indent=2, allow_nan=False) + '\n').encode())
    print(json.dumps(dict(status=record['status'], receipt=str(completion), sha256=sha(completion))))


if __name__ == '__main__':
    main()
