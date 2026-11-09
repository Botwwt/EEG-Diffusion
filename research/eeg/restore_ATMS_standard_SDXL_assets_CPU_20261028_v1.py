"""Restore pinned public standard-SDXL/IP assets; CPU only, no scientific run.

Run this entry on the existing server repository.  It delegates all network
transport/resume to the unchanged acquire_sdxl_base.acquire_snapshot.  Outputs
under /dev/shm are volatile RAM, not durable storage or disk expansion.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
import traceback
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
SERVER_ROOT = Path('/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1')
RAM = Path('/dev/shm/eeg_sota_autoresearch_20261018_v1')
SDXL = RAM / 'sdxl_base_46216598'
ADAPTER = RAM / 'ATMS_adapter_restore_20261028_v1'
ADAPTER_MEMBER = 'sdxl_models/ip-adapter_sdxl_vit-h.safetensors'
LOGICAL_ADAPTER = Path('/dev/shm/eeg_diffusion_next_20261018_v1/adapter.safetensors')
AUDIT = ROOT / 'audit/cross_framework_ordinary_20261028_v1'
MAX_SECONDS = 7200
SDXL_REVISION = '462165984030d82259a11f4367a4eed129e94a7b'
ADAPTER_REVISION = '018e402774aeeddd60609b4ecdb7e298259dc729'
ADAPTER_BYTES = 698391064
ADAPTER_SHA = 'ebf05d918348aec7abb02a5e9ecef77e0aaea6914a5c4ea13f50d45eb1681831'
PINS = {
    'acquire_sdxl_base.py': '5eef06bcada9a26df2e8aa8856008d2749a17ea70a312142baa9ae868801cc08',
    'audit/sdxl_base_selected_assets.json': 'a52b4c5211ae5f55c01d152d799d864193944811d55271b45c36a897ed7a51d4',
}
ACTIVE = None


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def plain(path):
    """Reject symlinks in the exact file and all existing ancestors."""
    path = Path(path)
    assert path.is_absolute() and '..' not in path.parts, path
    for item in (path, *path.parents):
        assert not item.is_symlink(), f'No symlink adoption: {item}'
    return path


def write_new(path, value):
    path = plain(path)
    raw = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')
    assert len(raw) <= 256 << 10, 'Bounded metadata'
    # Exclusive output preserves all earlier success/failure receipts.
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def manifests():
    for name, expected in PINS.items():
        path = plain(ROOT / name)
        assert path.is_file() and sha256(path) == expected, f'Pinned source/input changed: {name}'
    base = json.loads((ROOT / 'audit/sdxl_base_selected_assets.json').read_text(encoding='utf-8'))
    assert base['repository'] == 'stabilityai/stable-diffusion-xl-base-1.0'
    assert base['revision'] == SDXL_REVISION and len(base['assets']) == 18
    adapter = dict(repository='h94/IP-Adapter', revision=ADAPTER_REVISION, assets=[dict(
        path=ADAPTER_MEMBER, size=ADAPTER_BYTES, lfs=dict(oid=ADAPTER_SHA, size=ADAPTER_BYTES))])
    return {'sdxl': (base, SDXL), 'adapter': (adapter, ADAPTER)}


def safe_output(manifest, output):
    assert output.is_relative_to(RAM) and output != RAM
    plain(output).mkdir(parents=True, exist_ok=True)
    for asset in manifest['assets']:
        relative = Path(asset['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        target = output / relative
        for path in (target, target.with_name(target.name + '.download'),
                     target.with_name(target.name + '.chunks.jsonl')):
            plain(path)
            assert not path.exists() or path.is_file(), path
    for name in ('acquisition.lock', 'progress.json', 'completion.json', 'failure.json'):
        path = plain(output / name)
        assert not path.exists() or path.is_file(), path


def worker(kind, parent_pid):
    assert ROOT == SERVER_ROOT, 'Server entry only'
    assert parent_pid == os.getppid(), 'Internal child must have its live restoration parent'
    available = manifests()
    manifest, output = available[kind]
    safe_output(manifest, output)
    # Original module, original chunk/journal/full-SHA validation and RAM guard.
    import acquire_sdxl_base as M
    assert Path(M.__file__).resolve() == ROOT / 'acquire_sdxl_base.py'
    M.acquire_snapshot(manifest, str(output))


def stop_child(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def run_snapshot(kind, token, work_deadline):
    global ACTIVE
    remaining = work_deadline - time.monotonic()
    assert remaining > 0, 'Finite whole-restoration deadline'
    stdout_path = plain(AUDIT / f'{token}_{kind}.stdout.log')
    stderr_path = plain(AUDIT / f'{token}_{kind}.stderr.log')
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    started = time.monotonic()
    with stdout_path.open('xb') as out, stderr_path.open('xb') as err:
        ACTIVE = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
            '--worker', kind, '--parent-pid', str(os.getpid())], stdin=subprocess.DEVNULL,
            stdout=out, stderr=err, cwd=str(ROOT), env=environment,
            start_new_session=True, close_fds=True)
        pid = ACTIVE.pid
        try:
            code = ACTIVE.wait(timeout=remaining)
        except BaseException:
            stop_child(ACTIVE)
            raise
        finally:
            if ACTIVE.poll() is not None:
                ACTIVE = None
    assert code == 0, f'{kind} acquisition failed with exit {code}; preserve logs and download/chunks'
    return dict(kind=kind, pid=pid, exit_code=code, seconds=time.monotonic() - started,
        stdout=str(stdout_path), stderr=str(stderr_path))


def verified_files(manifest, output):
    # Reuse the exact original SHA1 git-blob / SHA256 LFS identities, then record
    # full SHA256 for all nineteen files in the durable receipt.
    import acquire_sdxl_base as M
    result = []
    for asset in manifest['assets']:
        target = plain(output / asset['path'])
        assert target.is_file() and stat.S_ISREG(target.stat().st_mode)
        before = target.stat()
        M.verify(target, asset)
        digest = sha256(target)
        after = target.stat()
        assert (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), target
        result.append(dict(path=str(target), bytes=after.st_size, sha256=digest,
            original_identity=asset.get('lfs', {}).get('oid', asset.get('oid')),
            original_identity_algorithm='sha256' if 'lfs' in asset else 'git-blob-sha1',
            original_identity_verified=True))
    return result


def link_adapter():
    source = plain(ADAPTER / ADAPTER_MEMBER)
    assert source.is_file() and source.stat().st_size == ADAPTER_BYTES and sha256(source) == ADAPTER_SHA
    target = plain(LOGICAL_ADAPTER)
    target.parent.mkdir(parents=True, exist_ok=True)
    assert target.parent.stat().st_dev == source.stat().st_dev, 'Hardlink must remain on the same RAM filesystem'
    created = False
    if not target.exists():
        try:
            os.link(source, target, follow_symlinks=False)
            created = True
        except FileExistsError:
            pass  # A concurrent creator is verified below; never overwrite it.
    plain(target)
    assert target.is_file() and target.stat().st_size == ADAPTER_BYTES and sha256(target) == ADAPTER_SHA, (
        'Existing adapter differs; refusing replacement')
    return dict(source=str(source), logical_path=str(target), created_hardlink=created,
        same_inode=os.path.samefile(source, target), bytes=ADAPTER_BYTES, sha256=ADAPTER_SHA,
        existing_same_sha_file_accepted=not created)


def main():
    global ACTIVE
    assert ROOT == SERVER_ROOT, 'Server entry only; do not download public weights locally'
    started = time.monotonic()
    started_utc = utc()
    work_deadline = started + MAX_SECONDS - 15  # Reserve shutdown/receipt time.
    def timeout(_signum, _frame):
        raise TimeoutError('Entire CPU asset restoration reached the fixed 7200-second limit')
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, MAX_SECONDS)
    plain(AUDIT).mkdir(parents=True, exist_ok=True)
    # RAM location is asserted before any downloader execution; metadata is durable.
    assert plain(Path('/dev/shm')).is_dir()
    assert ROOT.stat().st_dev != Path('/dev/shm').stat().st_dev, 'Audit must be persistent'
    token = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_' + uuid.uuid4().hex[:8]
    lock_path = plain(AUDIT / 'public_assets_restore.lock')
    with lock_path.open('a') as lock:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        evidence = dict(started_utc=started_utc, pid=os.getpid(), root=str(ROOT),
            roots=dict(sdxl=str(SDXL), adapter_restore=str(ADAPTER), adapter_logical=str(LOGICAL_ADAPTER),
                persistent_audit=str(AUDIT)), source_sha256=dict(PINS),
            entry_sha256=sha256(Path(__file__).resolve()), max_seconds=MAX_SECONDS,
            GPU_jobs=0, model_forwards=0, training_updates=0, generated_images=0,
            storage='All restored public weights are volatile /dev/shm RAM; not persistent disk expansion',
            existing_images_weights_and_download_chunks_deleted=0, stages=[])
        try:
            available = manifests()
            for manifest, output in available.values():
                safe_output(manifest, output)
            # Refuse a mismatched existing consumer path before any download.
            if plain(LOGICAL_ADAPTER).exists():
                assert LOGICAL_ADAPTER.is_file() and LOGICAL_ADAPTER.stat().st_size == ADAPTER_BYTES
                assert sha256(LOGICAL_ADAPTER) == ADAPTER_SHA, 'Existing adapter differs; never replace'
            for kind in ('sdxl', 'adapter'):
                evidence['stages'].append(run_snapshot(kind, token, work_deadline))
            evidence['snapshots'] = [dict(repository=manifest['repository'], revision=manifest['revision'],
                root=str(output), files=verified_files(manifest, output))
                for manifest, output in available.values()]
            evidence['adapter_link'] = link_adapter()
            evidence.update(status='PINNED_STANDARD_SDXL_AND_H94_ADAPTER_RESTORED_CPU_SHA_VERIFIED',
                completed_utc=utc(), seconds=time.monotonic() - started, scientific_result=False)
            canonical = plain(AUDIT / 'public_assets_restored.json')
            if canonical.exists():
                old = json.loads(canonical.read_text(encoding='utf-8'))
                assert old['status'] == evidence['status'] and old['source_sha256'] == evidence['source_sha256']
                previous = AUDIT / f'public_assets_restored_previous_{token}.json'
                os.link(canonical, previous, follow_symlinks=False)
                evidence['previous_canonical_receipt_preserved'] = str(previous)
            result_path = AUDIT / f'public_assets_restored_{token}.json'
            write_new(result_path, evidence)
            # New immutable receipt and prior canonical are both retained.
            pending = AUDIT / f'public_assets_restored_{token}.pending'
            os.link(result_path, pending, follow_symlinks=False)
            os.replace(pending, canonical)
            print(json.dumps(evidence, allow_nan=False), flush=True)
        except BaseException as exc:
            stop_child(ACTIVE)
            ACTIVE = None
            evidence.update(status='FAILED_RESUMABLE_DOWNLOADS_CHUNKS_AND_EXISTING_FILES_PRESERVED',
                failed_utc=utc(), seconds=time.monotonic() - started, type=type(exc).__name__,
                message=str(exc), traceback=traceback.format_exc(), scientific_result=False)
            write_new(AUDIT / f'public_assets_restore_failure_{token}.json', evidence)
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=('sdxl', 'adapter'), help=argparse.SUPPRESS)
    parser.add_argument('--parent-pid', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.parent_pid)
    else:
        assert args.parent_pid is None
        main()
