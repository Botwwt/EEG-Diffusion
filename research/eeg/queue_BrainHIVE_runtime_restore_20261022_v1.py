"""Finite continuation from public download to original T-cache restoration.

Never admits or launches functional-v2 fitting. A failed producer is preserved,
and the queue stops instead of inventing readiness or retrying indefinitely.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'audit/runtime_restore_queue_20261022_v1.json'
MINIMUM_FORMAL_BYTES = 26655048192


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(value):
    temporary = OUT.with_name(OUT.name + '.pending')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(OUT)


def alive(pid):
    path = Path('/proc') / str(pid) / 'cmdline'
    return path.exists() and bool(path.read_bytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--public-download-pid', required=True, type=int)
    args = parser.parse_args()
    assert os.name == 'posix' and ROOT == Path('/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1')
    assert not OUT.exists()
    started = time.monotonic()
    public_path = ROOT / 'audit/runtime_public_restore_20261022_v1.json'
    state = dict(status='WAITING_FOR_EXACT_PUBLIC_RUNTIME_ASSETS',
        start_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), pid=os.getpid(),
        public_download_pid=args.public_download_pid, maximum_total_seconds=9600,
        source_sha256=sha(__file__), formal_functional_training_started=False,
        formal_storage_admitted=False, latest_GPU_job=None)
    write(state)
    try:
        while not public_path.exists():
            assert time.monotonic() - started < 7200, 'Finite public download wait ended'
            assert alive(args.public_download_pid), 'Public producer ended without exact closure'
            state.update(seconds=time.monotonic() - started,
                observation_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
            write(state)
            time.sleep(20)
        public = json.loads(public_path.read_text())
        assert public['status'] == 'ALL19_PINNED_PUBLIC_RUNTIME_ASSETS_NATIVE_SHA_VERIFIED'
        assert len(public['files']) == 19
        for row in public['files']:
            assert sha(row['path']) == row['sha256']
        for filename, expected_status in [
            ('runtime_input_containers_restored_20261022_v1.json', 'EXACT_METADATA_AND_ALL16540_JPEG_MEMBER_INPUTS_RESTORED'),
            ('runtime_30_condition_caches_restored_20261022_v1.json', 'ALL30_FULL_AND_OOF_CACHES_ORIGINAL_EXACT_SHA_RESTORED')]:
            receipt = json.loads((ROOT / 'audit' / filename).read_text())
            assert receipt['status'] == expected_status
            if 'files' in receipt:
                assert len(receipt['files']) == 30
                for row in receipt['files']:
                    assert sha(row['path']) == row['sha256']
        condition_job = json.loads((ROOT / 'audit/BrainHIVE_restore_existing_30_condition_caches_20261022_v1_job.json').read_text())
        assert condition_job['status'] == 'COMPLETED' and not alive(condition_job['pid'])
        name = 'BrainHIVE_restore_existing_T_latents_20261022_v1'
        log = ROOT / 'audit/runtime_T_latents_restore_20261022_v1.log'
        assert not log.exists() and not (ROOT / 'audit' / (name + '_job.json')).exists()
        state.update(status='RESTORING_ORIGINAL_T_LATENTS_NO_FITTING', latest_GPU_job=name,
                     public_closure_sha256=sha(public_path))
        write(state)
        with log.open('xb') as stream:
            result = subprocess.run(['/root/miniconda3/bin/python', '-u', str(ROOT / 'run_gpu_job_v2.py'),
                '--name', name, '--max-seconds', '1800', '--', '/root/miniconda3/bin/python', '-u',
                str(ROOT / 'restore_BrainHIVE_T_latents_GPU_20261022_v1.py')],
                stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT, timeout=1860)
        assert result.returncode == 0, 'T-only exact-cache restoration failed; preserve job and log'
        latent = json.loads((ROOT / 'audit/runtime_T_latents_exact_restored_20261022_v1.json').read_text())
        assert sha(latent['path']) == latent['sha256']
        recipe = json.loads((ROOT / 'configs/functional_profiles_20261022_v1/O.json').read_text())
        for path, digest in recipe['input_sha256'].items():
            assert sha(path) == digest, path
        for name, digest in recipe['source_sha256'].items():
            assert sha(ROOT / name) == digest, name
        free = {path: shutil.disk_usage(path).free for path in ('/root', '/root/autodl-tmp')}
        state.update(status='CORE_TRAINING_INPUTS_RESTORED_FORMAL_LIFECYCLE_NOT_ADMITTED',
            end_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            training_profile_inputs_SHA_verified=len(recipe['input_sha256']),
            all30_condition_caches_exact_SHA_verified=True, T_latent_exact_SHA_verified=True,
            physical_free=free, minimum_formal_four_arm_bytes=MINIMUM_FORMAL_BYTES,
            aggregate_minimum_space_gap=max(0, MINIMUM_FORMAL_BYTES - sum(free.values())),
            evaluation_runtime_asset_restoration_not_verified=True,
            core_inputs_ready=True, full_lifecycle_ready=False,
            formal_functional_training_started=False, formal_storage_admitted=False,
            seconds=time.monotonic() - started)
        write(state)
    except Exception as error:
        state.update(status='RESTORATION_PARTIAL_WITH_EXPLICIT_FAILURE_NO_CASCADING_TRAINING',
            error_type=type(error).__name__, error=str(error),
            end_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            seconds=time.monotonic() - started)
        write(state)
        raise


if __name__ == '__main__':
    main()
