"""One finite CPU delivery check; no training, images, scores or GPU imports."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'audit/CPU_contract_delivery_20261022_v1/server_execution_v2.json'
SOURCES = {
    'brainhive_official_encoder_job_contract_CPU_v1.py': '310f1246396720e476c7897f2d593cf82ac91ee573495e38d14fb102862304a5',
    'check_BrainHIVE_official_encoder_job_contract_CPU_v1.py': '95082a3f8be9be3aa161e71bcb41da1d67f01e9e6ed1cb4baf2b6979da3eb1c0',
    'brainhive_official_benchmark_contract_v1.py': '9965ae4f85959c12049689ca0d7c6795ba5b35f18aa0b14896410074957b3c48',
    'check_BrainHIVE_official_benchmark_contract_CPU_v1.py': '4f49b01bdc676905b744b2bd46325ed030630aaed91a7b22f1f7ae2aa892b224',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def headroom():
    memory = {line.split(':')[0]: int(line.split()[1]) * 1024
              for line in Path('/proc/meminfo').read_text().splitlines()
              if line.startswith('MemAvailable:')}
    current = int(Path('/sys/fs/cgroup/memory.current').read_text())
    maximum = int(Path('/sys/fs/cgroup/memory.max').read_text())
    value = min(memory['MemAvailable'], maximum - current)
    assert value >= 12 << 30, 'Leave current formal training its fixed memory protection'
    return value


def main():
    assert os.name == 'posix' and not OUTPUT.exists(), 'Retain every previous execution receipt'
    OUTPUT.parent.mkdir(exist_ok=True)
    frozen = json.loads((ROOT / 'configs/BrainHIVE_new_instance_continuation_sources_20261022_v2.json').read_text())['source_sha256']
    assert len(frozen) == 29
    pins = dict(frozen, **SOURCES)
    assert all(sha(ROOT / name) == value for name, value in pins.items())
    result_sha = sha(ROOT / 'results.tsv')
    assert result_sha == '2ee905cc31949671c3068f04c9dbd05046ed4ae9888b05c9ca7cb08d781fb010'
    started = time.monotonic()
    record = dict(status='CPU_FUTURE_OFFICIAL_CONTRACT_CHECK_RUNNING', UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        source_sha256=sha(__file__), checked_source_sha256=SOURCES, stages=[],
        actual_new_fits=0, actual_new_condition_tensors=0, actual_images_read_or_generated=0,
        actual_metric_values_used=False, actual_GPU_jobs=0, method_locked=False, fitting_admitted=False)
    commands = [[sys.executable, '-B', name] for name in (
        'check_BrainHIVE_official_encoder_job_contract_CPU_v1.py',
        'check_BrainHIVE_official_benchmark_contract_CPU_v1.py')]
    commands[0].extend(['--plan-directory', str(ROOT / 'audit/official_fullfit_metadata_plan_server_20261022_v1')])
    commands.append([sys.executable, '-B', 'brainhive_official_encoder_job_contract_CPU_v1.py',
        '--plan-directory', str(ROOT / 'audit/official_fullfit_metadata_plan_server_20261022_v1')])
    try:
        for command in commands:
            before = headroom()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
            step = time.monotonic()
            completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, timeout=60)
            row = dict(command=command, exit_code=completed.returncode, seconds=time.monotonic() - step,
                CPU_headroom_before_bytes=before, CPU_headroom_after_bytes=headroom(),
                stdout=completed.stdout.decode(errors='replace'), stderr=completed.stderr.decode(errors='replace'))
            record['stages'].append(row)
            assert len(completed.stdout) + len(completed.stderr) <= 65536, 'Unexpected output'
            assert completed.returncode == 0, 'CPU contract failed; preserve actual failure'
        assert all(sha(ROOT / name) == value for name, value in pins.items())
        assert sha(ROOT / 'results.tsv') == result_sha
        record.update(status='FUTURE_OFFICIAL_CPU_CONTRACTS_AND_REAL_FULL_PLAN_VERIFIED_NO_ADMISSION',
            unchanged_frozen_sources=29, unchanged_results_sha256=result_sha,
            encoder_checks=14, benchmark_synthetic_groups=13, actual_plan_tasks=20, actual_plan_future_queries=165400,
            actual_payloads_or_hooks_certified=False, automatic_official_launch_enabled=False)
    except BaseException as error:
        record.update(status='CPU_FUTURE_OFFICIAL_CONTRACT_CHECK_FAILED', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record['seconds'] = time.monotonic() - started
        OUTPUT.write_bytes((json.dumps(record, indent=2) + '\n').encode())
    print(json.dumps(dict(status=record['status'], receipt=str(OUTPUT), sha256=sha(OUTPUT))))


if __name__ == '__main__':
    main()
