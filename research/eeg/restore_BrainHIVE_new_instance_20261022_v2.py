"""Rebuild pinned runtime bytes on the replacement GPU and retain disk copies.

Previous receipts and sources stay immutable. Exact replay mismatches stop the
stage; this program never substitutes a tolerance, refits a model or starts v2.
"""
import argparse
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / 'audit/new_instance_restore_20261022_v2'
DURABLE = Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/runtime_durable_20261022_v2')
WORK = Path('/root/brainhive_reproduction_20261020_v1')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    pending = path.with_name(path.name + '.pending')
    with pending.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    pending.replace(path)


def preserve_runtime_files(records, stage):
    """Copy verified ephemeral files to the expanded private data filesystem."""
    assert DURABLE.resolve().is_relative_to('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')
    copied = []
    for row in records:
        original = Path(row['path']); source = original.resolve(strict=True)
        assert source.is_relative_to('/dev/shm') and source.is_file()
        assert sha(source) == row['sha256']
        target = DURABLE / 'rootfs' / source.relative_to('/')
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not target.exists():
            pending = target.with_name(target.name + '.pending')
            assert not pending.exists() and not target.is_symlink()
            assert shutil.disk_usage(DURABLE).free >= source.stat().st_size + (40 << 30)
            with source.open('rb') as inp, pending.open('xb') as out:
                shutil.copyfileobj(inp, out, 8 << 20); out.flush(); os.fsync(out.fileno())
            assert sha(pending) == row['sha256']; pending.rename(target)
        assert target.is_file() and not target.is_symlink() and sha(target) == row['sha256']
        copied.append(dict(original_path=str(original), RAM_path=str(source), durable_path=str(target),
                           sha256=row['sha256'], bytes=target.stat().st_size))
    receipt = dict(status='EXACT_RUNTIME_BYTES_RETAINED_ON_EXPANDED_PRIVATE_DISK', stage=stage,
        UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), files=copied,
        total_payload_bytes=sum(x['bytes'] for x in copied),
        result_or_checkpoint_deletions=0, formal_training_started=False)
    write(AUDIT / (stage + '_durable.json'), receipt)
    return receipt


def seed_image_container(module):
    """Recreate the retained JPEG members before checking an existing broken alias."""
    target = module.RAM / 'training_images_member_identical_v1.zip'
    if target.exists():
        return
    assert module.ORIGINAL.is_symlink() and module.ORIGINAL.resolve(strict=False) == target
    rows = [r for r in csv.DictReader((ROOT / 'audit/image_manifest.csv').open()) if r['dataset_split'] == 'train']
    assert len(rows) == 16540
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    pending = target.with_name(target.name + '.pending'); assert not pending.exists()
    with zipfile.ZipFile(pending, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for row in rows:
            relative = Path(row['image_member']).relative_to('training_images')
            assert '..' not in relative.parts
            raw = (module.JPEG_ROOT / relative).read_bytes()
            assert len(raw) == int(row['image_bytes']) and hashlib.sha256(raw).hexdigest() == row['image_sha256']
            info = zipfile.ZipInfo(row['image_member'], date_time=(1980, 1, 1, 0, 0, 0)); info.external_attr = 0o600 << 16
            z.writestr(info, raw)
    with pending.open('rb') as stream: os.fsync(stream.fileno())
    with zipfile.ZipFile(pending) as z:
        assert len(z.infolist()) == 16540 and z.testzip() is None
        assert all(hashlib.sha256(z.read(r['image_member'])).hexdigest() == r['image_sha256'] for r in rows)
    expected = json.loads((ROOT / 'audit/runtime_input_containers_restored_20261022_v1.json').read_text())
    assert sha(pending) == expected['container_sha256']
    pending.rename(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['public', 'inputs', 'conditions', 'latent', 'verify'], required=True)
    parser.add_argument('--workers', type=int, default=64)
    args = parser.parse_args()
    assert os.name == 'posix' and ROOT == Path('/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1')
    assert 1 <= args.workers <= 64
    AUDIT.mkdir(parents=True, exist_ok=True); DURABLE.mkdir(parents=True, exist_ok=True, mode=0o700)
    out = AUDIT / (args.stage + '.json')
    assert not out.exists(), 'Do not replace any previous stage or select among repeats'
    start = time.monotonic()
    pinned = json.loads((ROOT / 'configs/functional_profiles_20261022_v1/O.json').read_text())['source_sha256']
    assert all(sha(ROOT / p) == h for p, h in pinned.items())
    assert shutil.disk_usage(DURABLE).free > 60 << 30
    if args.stage == 'verify':
        recipe = json.loads((ROOT / 'configs/functional_profiles_20261022_v1/O.json').read_text())
        assert all(sha(p) == h for p, h in recipe['input_sha256'].items())
        for stage in ('public', 'inputs', 'conditions', 'latent'):
            proof = json.loads((AUDIT / (stage + '_durable.json')).read_text())
            for item in proof['files']:
                assert sha(item['RAM_path']) == sha(item['durable_path']) == item['sha256']
        write(out, dict(status='REPLACEMENT_INSTANCE_CORE_RUNTIME_AND_DURABLE_COPIES_EXACT_SHA_VERIFIED',
            UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), inputs=len(recipe['input_sha256']),
            source_sha256=sha(__file__), source_pins_unchanged=True, formal_training_started=False,
            storage_or_full_lifecycle_admission_issued=False, seconds=time.monotonic()-start))
        print(out.read_text(), flush=True); return
    modules = dict(public='restore_BrainHIVE_public_runtime_CPU_20261022_v1',
                   inputs='restore_BrainHIVE_input_containers_CPU_20261022_v1',
                   conditions='restore_BrainHIVE_conditions_GPU_20261022_v1',
                   latent='restore_BrainHIVE_T_latents_GPU_20261022_v1')
    module = importlib.import_module(modules[args.stage])
    original_source_sha = sha(module.__file__)
    module.OUT = out
    if hasattr(module, 'PROGRESS'): module.PROGRESS = AUDIT / (args.stage + '_progress.json')
    if args.stage == 'public':
        module.START = time.monotonic()
        sys.argv = [str(Path(module.__file__)), '--workers', str(args.workers)]
    elif args.stage == 'inputs':
        seed_image_container(module)
    elif args.stage == 'conditions':
        module.STAGING = Path('/dev/shm/BrainHIVE_private_derived_cache_20261020_v2/new_instance_condition_staging_v2')
    elif args.stage == 'latent':
        public = json.loads((AUDIT / 'public.json').read_text())
        assert public['status'] == 'ALL19_PINNED_PUBLIC_RUNTIME_ASSETS_NATIVE_SHA_VERIFIED'
        assert all(sha(x['path']) == x['sha256'] for x in public['files'])
    module.main()
    assert sha(module.__file__) == original_source_sha
    result = json.loads(out.read_text())
    if args.stage in ('public', 'conditions'): records = result['files']
    elif args.stage == 'latent': records = [dict(path=result['path'], sha256=result['sha256'])]
    else: records = result['metadata'] + [dict(path=result['original_manifest_path'], sha256=result['container_sha256'])]
    durable = preserve_runtime_files(records, args.stage)
    write(AUDIT / (args.stage + '_driver.json'), dict(source_sha256=sha(__file__),
        original_helper_sha256=original_source_sha, stage_receipt_sha256=sha(out),
        durable_receipt_sha256=sha(AUDIT / (args.stage + '_durable.json')),
        durable_files=len(durable['files']), seconds=time.monotonic()-start,
        fixed_scientific_sources_unchanged=True, cleanup_performed=False))
    assert all(sha(ROOT / p) == h for p, h in pinned.items())
    print(json.dumps(dict(stage=args.stage, durable_files=len(durable['files']),
        status='STAGE_COMPLETE_OLD_NATIVE_SHAS_AND_NEW_DURABLE_COPIES_VERIFIED')), flush=True)


if __name__ == '__main__':
    main()
