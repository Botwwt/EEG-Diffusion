"""Release clean page cache of verified private durable runtime copies only.

POSIX_FADV_DONTNEED does not remove files or change their contents. No global
drop_caches, SHM deletion, parameter update, or resource-threshold change occurs.
"""
from pathlib import Path
import hashlib
import json
import os
import stat
import time

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'audit/new_instance_restore_20261022_v2'
DURABLE = Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/runtime_durable_20261022_v2')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def ref(path):
    return dict(path=str(path), sha256=sha(path), bytes=Path(path).stat().st_size)


def memory():
    base = Path('/sys/fs/cgroup')
    values = dict(line.split() for line in (base / 'memory.stat').read_text().splitlines())
    return dict(max=int((base / 'memory.max').read_text()),
        current=int((base / 'memory.current').read_text()),
        **{key: int(values[key]) for key in ('shmem', 'file', 'active_file', 'inactive_file')},
        events={key: int(value) for key, value in (line.split() for line in (base / 'memory.events').read_text().splitlines())})


def advise(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        assert stat.S_ISREG(os.fstat(fd).st_mode)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def main():
    out = BASE / 'pagecache_release_20261022_v1.json'
    assert not out.exists(), 'Keep every prior execution receipt'
    closure_path = BASE / 'exact_coverage_closure_core_and_evaluation_attempt1.json'
    closure = json.loads(closure_path.read_bytes())
    assert closure['status'] == 'EXACT_ORIGINAL_IDENTITY_COVERAGE_RAM_AND_DURABLE_RUNTIME_CLOSURE_VERIFIED'
    assert closure['core_original_records'] == 55 and closure['evaluator_original_records'] == 25
    # A completed wrapper and empty CUDA process list are independently required
    # by the recovery controller before restart. This helper never launches work.
    failure_paths = dict(formal_queue=ROOT / 'audit/functional_full_lifecycle_20261022_v1/formal_queue.json',
        fit_job=ROOT / 'audit/BrainHIVE_functional_v2_O_expanded_lifecycle_20261022_v1_fit_job.json',
        fit_log=ROOT / 'audit/functional_full_lifecycle_20261022_v1/BrainHIVE_functional_v2_O_expanded_lifecycle_20261022_v1_fit.log')
    target = Path('/root/brainhive_reproduction_20261020_v1/components/functional_alignment_v2/O')
    failed_output_files = sorted(target.iterdir())
    assert {p.name for p in target.iterdir()} == {'input_provenance.json', 'launch_3551.json', 'training.jsonl', 'progress.json'}
    before = memory()
    unique = {}
    for stage in closure['stages']:
        for item in stage['files']:
            path = Path(item['durable_path'])
            assert path.is_absolute() and path.is_relative_to(DURABLE)
            assert path.resolve() == path and path.is_file() and not path.is_symlink()
            if str(path) in unique:
                assert unique[str(path)]['sha256'] == item['sha256']
            unique[str(path)] = item
    assert unique
    records = []
    calls = 0
    for name, item in unique.items():
        path = Path(name)
        initial = path.stat()
        assert initial.st_size == item['bytes']
        first = sha(path)
        assert first == item['sha256']
        advise(path)
        calls += 1
        second = sha(path)
        final = path.stat()
        assert first == second and (initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns) == (
            final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns)
        records.append(dict(path=name, bytes=initial.st_size, sha256_before=first, sha256_after=second,
            device=initial.st_dev, inode=initial.st_ino))
    # The integrity read itself repopulates clean cache. Release it after all
    # content checks, then measure actual rather than estimated cgroup headroom.
    for record in records:
        advise(record['path'])
        calls += 1
    time.sleep(1)
    after = memory()
    import brainhive_functional_training_contract_v2 as C
    effective = C.resource_headroom()['effective_headroom_bytes']
    assert after['events']['oom'] == before['events']['oom'] == 0
    assert after['events']['oom_kill'] == before['events']['oom_kill'] == 0
    result = dict(status='TARGETED_PRIVATE_RUNTIME_PAGECACHE_RELEASE_VERIFIED_NO_FILE_REMOVAL',
        UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), issued_epoch=time.time(), source_sha256=sha(__file__),
        original_runtime_closure=ref(closure_path), files=records, cgroup_before=before, cgroup_after=after,
        effective_headroom_after_bytes=effective, minimum_restart_headroom_bytes=24 << 30,
        restart_headroom_satisfied=effective >= 24 << 30, actual_fadvise_calls=calls,
        failure_evidence={**{key: ref(p) for key, p in failure_paths.items()},
            'output_files': [ref(p) for p in failed_output_files]},
        original_identity_closure=ref(closure_path),
        deletion_performed=False, quantization_performed=False, GPU_used=False,
        headroom_threshold_changed=False, original_formal_minimum_headroom_bytes=8 << 30,
        SHA_and_path_and_inode_unchanged=True)
    out.write_bytes((json.dumps(result, indent=2) + '\n').encode())
    print(json.dumps(dict(path=str(out), sha256=sha(out), files=len(records), calls=calls,
        headroom=effective, restart_headroom_satisfied=result['restart_headroom_satisfied'])))


if __name__ == '__main__':
    main()
