"""Free DATA blocks by preserving inactive derived features at private RAM links.

Original EEG, learned weights and historical images are never selected. RAM is
volatile; preserved sources permit regeneration but bit-identical recomputation
after a reboot has not been demonstrated. Existing bytes are verified exactly.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time

ROOT = Path(__file__).resolve().parent
DATA = Path('/root/autodl-tmp')
RAM = Path('/dev/shm/brainhive_retired_ATM_features_20261021_v1')
ALLOWED = {f'atm_encoder_{role}_seed0_20261017_v1' for role in
           ['holdout', 'reconstruction_unit', 'retrieval_raw', 'retrieval_unit']}
NAMES = {f'{split}_{kind}.npy' for split in ['fit', 'calibration', 'development']
         for kind in ['single_trials', 'waveform_mean']}
GIB = 1 << 30


def sha(path, evict=False):
    with Path(path).open('rb') as stream:
        value = hashlib.file_digest(stream, 'sha256').hexdigest()
        if evict:
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return value


def memory():
    return dict(current=int(Path('/sys/fs/cgroup/memory.current').read_text()),
                maximum=int(Path('/sys/fs/cgroup/memory.max').read_text()),
                shm_free=shutil.disk_usage('/dev/shm').free)


def identities(paths):
    result = {}
    for path in paths:
        s = path.stat()
        result[(os.major(s.st_dev), os.minor(s.st_dev), s.st_ino)] = str(path)
    return result


def active_uses(paths):
    known = identities(paths)
    uses = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            for fd in (proc / 'fd').iterdir():
                try:
                    s = fd.stat()
                    key = (os.major(s.st_dev), os.minor(s.st_dev), s.st_ino)
                    if key in known:
                        uses.append(dict(pid=int(proc.name), kind='fd', path=known[key]))
                except (FileNotFoundError, PermissionError):
                    pass
            for line in (proc / 'maps').read_text().splitlines():
                fields = line.split(None, 5)
                major, minor = [int(x, 16) for x in fields[3].split(':')]
                key = (major, minor, int(fields[4]))
                if key in known:
                    uses.append(dict(pid=int(proc.name), kind='mapping', path=known[key]))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return uses


def write(path, value):
    pending = path.with_name(path.name + '.pending')
    with pending.open('w') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    inventory = ROOT / 'audit/BrainHIVE_storage_recovery_kan_readonly_v1.json'
    all_files = json.loads(inventory.read_text())['files']
    files = [x for x in all_files if Path(x['path']).suffix == '.npy']
    assert len(files) == 24
    paths = [Path(x['path']) for x in files]
    assert len(set(paths)) == 24
    for item, path in zip(files, paths):
        assert path.parent.name in ALLOWED and path.name in NAMES
        assert path.parent.resolve() == DATA / path.parent.name
        s = path.lstat()
        assert stat.S_ISREG(s.st_mode) and s.st_nlink == 1
        assert s.st_size == item['bytes'] and s.st_mtime_ns == item['mtime_ns']
        assert sha(path, evict=True) == item['sha256']
        assert sha(item['checkpoint']) == item['checkpoint_expected_sha256']
        assert Path(item['manifest']).is_file()
    uses = active_uses(paths)
    assert not uses, uses
    total = sum(x['bytes'] for x in files)
    mem = memory()
    assert mem['maximum'] - mem['current'] - total >= 4 * GIB, mem
    assert mem['shm_free'] - total >= GIB, mem
    receipt = ROOT / 'audit/BrainHIVE_retired_ATM_RAM_relocation_20261021_v1.json'
    report = dict(status='DRY_RUN_ONLY', UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        inventory_sha256=sha(inventory), files=files, logical_bytes=total, before_memory=mem,
        before_DATA_free=shutil.disk_usage(DATA).free, moved=[], active_inode_uses=uses,
        raw_EEG_learned_weights_or_historical_images_deleted=0, RAM_is_persistent_backup=False,
        restoration='Copy verified RAM bytes back to each original path if needed. After RAM loss regenerate derived features from retained encoder/manifest/raw contracts; byte-identical recomputation unproven.')
    if not args.apply:
        print(json.dumps({k: v for k, v in report.items() if k != 'files'}), flush=True)
        return
    assert not receipt.exists() and not RAM.exists()
    RAM.mkdir(mode=0o700)
    write(receipt, report)
    try:
        for item, source in zip(files, paths):
            size = item['bytes']
            mem = memory()
            assert mem['maximum'] - mem['current'] - 2 * size >= 4 * GIB, mem
            assert mem['shm_free'] - size >= GIB, mem
            assert not active_uses([source])
            previous = source.lstat()
            target = RAM / source.parent.name / source.name
            target.parent.mkdir(mode=0o700, exist_ok=True)
            assert not target.exists()
            pending = target.with_name(target.name + '.pending')
            digest = hashlib.sha256()
            with source.open('rb') as inp, pending.open('xb') as out:
                os.chmod(pending, 0o600)
                while chunk := inp.read(1 << 20):
                    out.write(chunk)
                    digest.update(chunk)
                out.flush()
                os.fsync(out.fileno())
                os.posix_fadvise(inp.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            assert digest.hexdigest() == sha(pending) == item['sha256']
            current = source.lstat()
            assert (current.st_ino, current.st_size, current.st_mtime_ns) == (
                previous.st_ino, previous.st_size, previous.st_mtime_ns)
            pending.replace(target)
            link = source.with_name(source.name + '.RAM_link_pending_20261021_v1')
            assert not link.exists() and not link.is_symlink()
            os.symlink(str(target), link)
            os.replace(link, source)
            assert source.is_symlink() and sha(source) == item['sha256']
            report['moved'].append(dict(original=str(source), RAM_target=str(target),
                sha256=item['sha256'], bytes=size, old_allocated_bytes=previous.st_blocks * 512,
                checkpoint=item['checkpoint'], checkpoint_sha256=item['checkpoint_expected_sha256'],
                manifest=item['manifest']))
            report['status'] = 'COPY_VERIFIED_ORIGINAL_PATH_LINKED_PROGRESS'
            write(receipt, report)
        from brainhive_lifecycle_storage_v1 import verify_remaining
        recipe = json.loads((ROOT / 'configs/BrainHIVE_bounded_main_recipe_v1.json').read_text())
        funded = verify_remaining(recipe)
        mem = memory()
        assert mem['maximum'] - mem['current'] >= 4 * GIB and mem['shm_free'] >= GIB, mem
        report.update(status='ALL24_BYTE_IDENTICAL_LINKS_AND_EXISTING_MAIN_LIFECYCLE_FUNDED',
            after_memory=mem, after_DATA_free=shutil.disk_usage(DATA).free,
            actual_DATA_free_increase=shutil.disk_usage(DATA).free - report['before_DATA_free'],
            main_remaining_forecast=funded)
        write(receipt, report)
        print(json.dumps({k: v for k, v in report.items() if k not in ['files', 'moved']}), flush=True)
    except BaseException as error:
        report.update(status='PARTIAL_RELOCATION_PRESERVED', error_type=type(error).__name__, error=str(error)[:1500])
        write(receipt, report)
        raise


if __name__ == '__main__':
    main()
