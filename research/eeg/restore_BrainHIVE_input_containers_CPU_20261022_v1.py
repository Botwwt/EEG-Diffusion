"""Restore exact archived metadata and a pixel-identical image input container.

The retired duplicate ZIP's container bytes are not claimed to be reproduced;
every original JPEG member must match the existing stimulus manifest instead.
"""
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import zipfile

ROOT = Path(__file__).resolve().parent
RAM = Path('/dev/shm/BrainHIVE_private_derived_cache_20261020_v2/restored_input_containers')
ORIGINAL = Path('/root/autodl-tmp/mced_research_bundle/external/things_eeg2_direct_v1/training_images.zip')
JPEG_ROOT = Path('/root/autodl-tmp/avde_single_task_data_20261016/image_set/training_images')
OUT = ROOT / 'audit/runtime_input_containers_restored_20261022_v1.json'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    assert os.name == 'posix' and str(ROOT).startswith('/root/atm_diffusion_20261015/research/')
    assert not OUT.exists()
    started = time.monotonic()
    RAM.mkdir(mode=0o700, parents=True, exist_ok=True)
    restored = []
    receipt = json.loads((ROOT / 'audit/BrainHIVE_storage_metadata_exact_archive_v14.json').read_text())
    for item in receipt['files']:
        original = Path(item['path']); target = Path(item['RAM_exact_alias'])
        assert target.is_relative_to('/dev/shm') and original.is_symlink()
        assert original.resolve(strict=False) == target
        if not target.exists():
            archive = Path(item['archive']); assert sha(archive) == item['archive_sha256']
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.restore.pending')
            assert not temporary.exists()
            with gzip.open(archive, 'rb') as source, temporary.open('xb') as destination:
                shutil.copyfileobj(source, destination, 4 << 20)
                destination.flush(); os.fsync(destination.fileno())
            assert temporary.stat().st_size == item['bytes'] and sha(temporary) == item['sha256']
            temporary.rename(target)
        assert sha(original) == item['sha256']
        restored.append(dict(path=str(original), sha256=item['sha256'], exact_bytes=True))
    rows = [row for row in csv.DictReader((ROOT / 'audit/image_manifest.csv').open())
            if row['dataset_split'] == 'train']
    assert len(rows) == len({row['image_member'] for row in rows}) == 16540
    assert all(row['image_zip'] == str(ORIGINAL) for row in rows)
    total = sum(int(row['image_bytes']) for row in rows)
    target = RAM / 'training_images_member_identical_v1.zip'
    assert shutil.disk_usage('/dev/shm').free > total + (512 << 20)
    if not target.exists():
        temporary = target.with_name(target.name + '.pending'); assert not temporary.exists()
        with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for row in rows:
                relative = Path(row['image_member']).relative_to('training_images')
                assert '..' not in relative.parts
                path = JPEG_ROOT / relative
                data = path.read_bytes()
                assert len(data) == int(row['image_bytes'])
                assert hashlib.sha256(data).hexdigest() == row['image_sha256']
                info = zipfile.ZipInfo(row['image_member'], date_time=(1980, 1, 1, 0, 0, 0))
                info.external_attr = 0o600 << 16
                archive.writestr(info, data)
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        with zipfile.ZipFile(temporary) as archive:
            assert len(archive.infolist()) == 16540 and archive.testzip() is None
            for row in rows:
                assert hashlib.sha256(archive.read(row['image_member'])).hexdigest() == row['image_sha256']
        temporary.rename(target)
    with zipfile.ZipFile(target) as archive:
        assert len(archive.infolist()) == 16540
        for row in rows:
            assert hashlib.sha256(archive.read(row['image_member'])).hexdigest() == row['image_sha256']
    if not ORIGINAL.exists():
        assert not ORIGINAL.is_symlink(), 'Do not replace an unknown broken link'
        ORIGINAL.symlink_to(target)
    assert ORIGINAL.is_symlink() and ORIGINAL.resolve(strict=True) == target
    value = dict(status='EXACT_METADATA_AND_ALL16540_JPEG_MEMBER_INPUTS_RESTORED',
        UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), source_sha256=sha(__file__),
        metadata=restored, image_manifest_sha256=sha(ROOT / 'audit/image_manifest.csv'),
        original_manifest_path=str(ORIGINAL), RAM_container=str(target),
        container_sha256=sha(target), container_bytes=target.stat().st_size,
        images_member_SHA_verified=16540, original_JPEG_bytes_changed=False,
        original_retired_ZIP_container_bytes_reproduced=False,
        durable_JPEGs_preserved=True, RAM_is_volatile=True,
        GPU_calls=0, fitting_or_scoring_calls=0, seconds=time.monotonic() - started)
    with OUT.open('x') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    main()
