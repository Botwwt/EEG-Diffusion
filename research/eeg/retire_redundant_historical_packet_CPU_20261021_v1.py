"""Retire one transport ZIP only after all original member bytes stay durable.

Default is a read-only plan printed to stdout.  --apply preserves missing or
volatile member copies in new private persistent storage before unlinking only
the fixed ZIP.  Member contents and metadata survive; byte-identical recovery
of the original ZIP container is not promised.  No GPU or RAM relocation.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import time
import zipfile

sys.dont_write_bytecode = True


ROOT = Path(__file__).resolve().parent
DATA = Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')
BASE = Path('/root/autodl-tmp/avde_rime_oracle_latent_20261015')
ARCHIVE = BASE / 'completed_nll_packet_v1.zip'
ARCHIVE_BYTES = 260212525
ARCHIVE_SHA = 'e56842bc1c38cee04175e589fb7152af776a379e7c72014e6a288ac12e2cd545'
BACKUP = DATA / 'private_history_packet_members_20261021_v1'
RECEIPT = ROOT / 'audit/BrainHIVE_historical_packet_member_retirement_20261021_v1.json'
OLD_PROOF = ROOT / 'audit/duplicate_transport_packet_retirement_20261020_v1.json'
OLD_MANIFEST = ROOT / 'audit/completed_nll_packet_v1.zip.packet_manifest.preserved_20261020_v1.json'
RECIPE = ROOT / 'configs/BrainHIVE_bounded_main_recipe_v1.json'
MARGIN = 64 << 20


def digest(stream):
    return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def file_sha(path):
    path = Path(path)
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        result = digest(stream)
        assert identity(os.fstat(stream.fileno())) == identity(before), 'File changed while hashing'
        if hasattr(os, 'posix_fadvise'):
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    assert identity(path.stat()) == identity(before), 'Path was replaced while hashing'
    return result


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def exclusive_bytes(path, value):
    with path.open('xb') as stream:
        os.chmod(path, 0o600)
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_receipt(value):
    temporary = RECEIPT.with_name(RECEIPT.name + '.pending')
    with temporary.open('xb') as stream:
        stream.write(encoded(value)); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, RECEIPT)
    fsync_directory(RECEIPT.parent)


def root_spec():
    recipe = json.loads(RECIPE.read_text())
    storage = json.loads(Path(recipe['storage_lifecycle_receipt']).read_text())
    roots = [Path(p).resolve(strict=True) for p in storage['allowed_private_roots']]
    devices = {p.stat().st_dev for p in roots}
    assert len(roots) == len(devices) == 2
    assert DATA.resolve(strict=True).is_relative_to('/root') and DATA.stat().st_dev in devices
    return recipe, roots, devices


def forecast(write_bytes):
    from brainhive_lifecycle_storage_v1 import verify_remaining
    recipe, roots, _ = root_spec()
    current = verify_remaining(recipe)
    matches = [i for i, root in enumerate(roots) if root.stat().st_dev == DATA.stat().st_dev]
    assert len(matches) == 1
    index = matches[0]
    free = shutil.disk_usage(DATA).free
    required = current['remaining_required'][index]
    assert free - write_bytes >= required + MARGIN, dict(
        reason='Preserve the full existing experiment and additional64MiB margin',
        DATA_free=free, planned_new_bytes=write_bytes, existing_remaining_need=required, extra_margin=MARGIN)
    return dict(main=current, DATA_free=free, DATA_remaining_need=required,
                planned_new_bytes=write_bytes, preserved_extra_margin=MARGIN)


def durable_candidate(source, expected_bytes, expected_sha, devices, archive_identity):
    try:
        original = Path(source)
        if not original.is_absolute():
            return None, 'Nonabsolute source or git pseudo-reference'
        resolved = original.resolve(strict=True)
        if not resolved.is_relative_to('/root') or resolved.is_relative_to('/proc') or resolved.is_relative_to('/dev/shm'):
            return None, 'Outside allowed private persistent paths; RAM aliases are not durable'
        info = resolved.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_dev not in devices:
            return None, 'Not a regular file on either verified private persistent filesystem'
        if (info.st_dev, info.st_ino) == archive_identity:
            return None, 'A ZIP container alias cannot replace its members'
        if info.st_size != expected_bytes or file_sha(resolved) != expected_sha:
            return None, 'Source has different historical bytes'
        return str(resolved), None
    except (FileNotFoundError, NotADirectoryError, PermissionError, OSError) as error:
        return None, type(error).__name__ + ': ' + str(error)[:240]


def member_metadata(info):
    return dict(filename=info.filename, date_time=list(info.date_time), compress_type=info.compress_type,
        compressed_bytes=info.compress_size, bytes=info.file_size, CRC32=info.CRC,
        flag_bits=info.flag_bits, create_system=info.create_system, create_version=info.create_version,
        extract_version=info.extract_version, internal_attr=info.internal_attr, external_attr=info.external_attr,
        volume=info.volume, header_offset=info.header_offset, extra_hex=info.extra.hex(), comment_hex=info.comment.hex())


def active_archive_uses():
    target = ARCHIVE.stat()
    uses = []
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            for fd in (process / 'fd').iterdir():
                try:
                    entry = fd.stat()
                    if (entry.st_dev, entry.st_ino) == (target.st_dev, target.st_ino):
                        uses.append(dict(pid=int(process.name), kind='fd'))
                except (FileNotFoundError, ProcessLookupError):
                    continue
            for line in (process / 'maps').read_text().splitlines():
                fields = line.split(None, 5)
                major, minor = (int(x, 16) for x in fields[3].split(':'))
                if (major, minor, int(fields[4])) == (os.major(target.st_dev), os.minor(target.st_dev), target.st_ino):
                    uses.append(dict(pid=int(process.name), kind='mmap'))
        except (FileNotFoundError, ProcessLookupError):
            continue
        # PermissionError intentionally propagates: no unverifiable idle claim.
    return uses


def plan():
    assert BASE.resolve(strict=True) == BASE and ARCHIVE.parent == BASE
    original = ARCHIVE.lstat()
    assert stat.S_ISREG(original.st_mode) and original.st_nlink == 1 and original.st_size == ARCHIVE_BYTES
    assert file_sha(ARCHIVE) == ARCHIVE_SHA
    assert not RECEIPT.exists() and not RECEIPT.with_name(RECEIPT.name + '.pending').exists()
    assert not BACKUP.exists() and not BACKUP.is_symlink(), 'Preserve any previous partial member backup'
    _, _, devices = root_spec()
    records, missing = [], {}
    with zipfile.ZipFile(ARCHIVE) as archive:
        infos = archive.infolist(); names = [i.filename for i in infos]
        assert len(names) == len(set(names)) == 76 and 'packet_manifest.json' in names
        raw_manifest = archive.read('packet_manifest.json')
        assert OLD_MANIFEST.read_bytes() == raw_manifest
        manifest = json.loads(raw_manifest)
        entries = manifest['files']
        assert len(entries) == len({r['archive_path'] for r in entries}) == 75
        assert set(names) == {r['archive_path'] for r in entries} | {'packet_manifest.json'}
        for entry in entries:
            member = entry['archive_path']; count = entry['bytes']; expected = entry['sha256']
            assert isinstance(count, int) and count >= 0 and len(expected) == 64
            assert all(char in '0123456789abcdef' for char in expected)
            info = archive.getinfo(member)
            assert not info.is_dir() and info.file_size == count
            with archive.open(info) as stream:
                assert digest(stream) == expected, ('ZIP member SHA mismatch', member)
            replacement, reason = durable_candidate(entry['source'], count, expected, devices,
                                                    (original.st_dev, original.st_ino))
            if replacement is None:
                if expected in missing:
                    assert missing[expected]['bytes'] == count
                else:
                    missing[expected] = dict(bytes=count, representative_member=member)
                replacement = str(BACKUP / 'blobs' / expected)
            records.append(dict(archive_path=member, original_source=entry['source'], bytes=count, sha256=expected,
                persistent_replacement=replacement, requires_new_blob=reason is not None,
                original_source_rejection=reason, zip_member_metadata=member_metadata(info)))
        metadata = dict(original_archive=str(ARCHIVE), original_archive_sha256=ARCHIVE_SHA,
            original_archive_bytes=ARCHIVE_BYTES, original_zip_comment_hex=archive.comment.hex(),
            manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(),
            manifest_zip_metadata=member_metadata(archive.getinfo('packet_manifest.json')),
            files=records, original_container_byte_identity_rebuild_proven=False,
            retained_contract='All original uncompressed member bytes and member metadata; original transport container retired')
    assert identity(ARCHIVE.lstat()) == identity(original)
    metadata_bytes = encoded(metadata)
    # Receipt reserve is explicit; never rely on eventual ZIP deletion for space.
    write_bytes = sum(item['bytes'] for item in missing.values()) + len(raw_manifest) + len(metadata_bytes) + (256 << 10)
    space = forecast(write_bytes)
    uses = active_archive_uses()
    assert not uses, uses
    result = dict(status='READ_ONLY_PLAN_NO_FILES_CHANGED', archive=str(ARCHIVE), archive_bytes=ARCHIVE_BYTES,
        archive_sha256=ARCHIVE_SHA, source_sha256=file_sha(__file__), old_proof_sha256=file_sha(OLD_PROOF),
        raw_manifest_sha256=metadata['manifest_sha256'], original_manifest_entries=75, ZIP_members_including_manifest=76,
        existing_durable_member_matches=sum(not row['requires_new_blob'] for row in records),
        members_requiring_new_persistent_copy=sum(row['requires_new_blob'] for row in records),
        unique_missing_blobs=len(missing), new_blob_bytes=sum(item['bytes'] for item in missing.values()),
        predicted_total_new_write_bytes=write_bytes, predicted_minimum_net_reclaimed_bytes=ARCHIVE_BYTES - write_bytes,
        backup_root=str(BACKUP), receipt=str(RECEIPT), before_space=space, archive_active_fd_or_mmap_uses=uses,
        GPU_used=False, RAM_relocation=False, original_container_byte_identity_rebuild_proven=False,
        raw_EEG_learned_weights_original_images_and_project_source_deleted=0)
    return result, metadata, metadata_bytes, raw_manifest, missing, original


def apply():
    lock_path = RECEIPT.with_suffix('.lock')
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result, metadata, metadata_bytes, raw_manifest, missing, original = plan()
        assert result['predicted_minimum_net_reclaimed_bytes'] > 0, 'Preservation must reclaim persistent storage'
        assert BACKUP.parent.resolve(strict=True) == DATA.resolve(strict=True)
        BACKUP.mkdir(mode=0o700)
        (BACKUP / 'blobs').mkdir(mode=0o700)
        result.update(status='PRESERVING_ALL_ORIGINAL_MEMBERS_ZIP_STILL_PRESENT',
            started_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        write_receipt(result)
        exclusive_bytes(BACKUP / 'packet_manifest.json', raw_manifest)
        exclusive_bytes(BACKUP / 'member_metadata.json', metadata_bytes)
        remaining = sum(item['bytes'] for item in missing.values())
        with zipfile.ZipFile(ARCHIVE) as archive:
            for expected, item in missing.items():
                forecast(remaining + (256 << 10))
                target = BACKUP / 'blobs' / expected
                with archive.open(item['representative_member']) as inp, target.open('xb') as out:
                    os.chmod(target, 0o600)
                    shutil.copyfileobj(inp, out, 1 << 20)
                    out.flush(); os.fsync(out.fileno())
                assert target.stat().st_size == item['bytes'] and file_sha(target) == expected
                remaining -= item['bytes']
        _, _, devices = root_spec()
        for row in metadata['files']:
            replacement, reason = durable_candidate(row['persistent_replacement'], row['bytes'], row['sha256'],
                devices, (original.st_dev, original.st_ino))
            assert replacement is not None and reason is None, ('Final persistent member validation failed', row['archive_path'])
        assert (BACKUP / 'packet_manifest.json').read_bytes() == raw_manifest
        assert file_sha(BACKUP / 'member_metadata.json') == hashlib.sha256(metadata_bytes).hexdigest()
        assert file_sha(ARCHIVE) == ARCHIVE_SHA and identity(ARCHIVE.lstat()) == identity(original)
        assert not active_archive_uses(), 'Keep the ZIP while another process has it open or mapped'
        forecast(256 << 10)
        # Persist each level's new directory entries before making the old
        # container removable. Every new file's data was already fsynced.
        for directory in [BACKUP / 'blobs', BACKUP, DATA]:
            fsync_directory(directory)
        result.update(status='ALL75_MEMBER_BYTES_AND_RAW_MANIFEST_DURABLE_VERIFIED_ZIP_STILL_PRESENT',
            durable_members_freshly_verified=75, raw_manifest_freshly_verified=True,
            new_blob_and_backup_parent_directories_fsynced=True,
            metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
            before_container_unlink_free_bytes=shutil.disk_usage(DATA).free)
        write_receipt(result)
        # Only this literal, reverified regular file is removed.  No source,
        # learned model, data directory, member backup or history is deleted.
        assert ARCHIVE == BASE / 'completed_nll_packet_v1.zip' and not ARCHIVE.is_symlink()
        assert identity(ARCHIVE.lstat()) == identity(original) and not active_archive_uses()
        ARCHIVE.unlink()
        fsync_directory(BASE)
        assert not ARCHIVE.exists()
        result.update(status='REDUNDANT_ZIP_CONTAINER_RETIRED_ALL_ORIGINAL_MEMBER_BYTES_PRESERVED',
            completed_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            after_container_unlink_free_bytes=shutil.disk_usage(DATA).free,
            removed_only_transport_container=True, container_parent_directory_fsynced=True)
        try:
            result['after_main_forecast'] = forecast(0)
        except Exception as error:
            result['after_main_forecast_error'] = type(error).__name__ + ': ' + str(error)[:1600]
        write_receipt(result)
        print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if args.apply:
        apply()
    else:
        result, metadata, _, _, _, _ = plan()
        print(json.dumps(result | dict(members_to_preserve=[
            row for row in metadata['files'] if row['requires_new_blob']])), flush=True)


if __name__ == '__main__':
    main()
