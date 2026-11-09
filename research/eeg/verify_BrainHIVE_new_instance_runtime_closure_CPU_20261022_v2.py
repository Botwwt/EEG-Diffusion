"""Read-only exact-coverage validation of replacement-instance runtime recovery.

Reads original identity contracts and actual stage/driver/copy receipts. It
writes one new bounded validation receipt, never changes runtime aliases,
historical proofs, images, scores, learned weights, or admission decisions.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
SERVER_ROOT = Path('/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1')
WORK = Path('/root/brainhive_reproduction_20261020_v1')
DATA = Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')
AUDIT = ROOT / 'audit/new_instance_restore_20261022_v2'
DURABLE = DATA / 'runtime_durable_20261022_v2'
COUNTS = dict(public=19, inputs=5, conditions=30, latent=1, evaluation=25)
STAGE_STATUS = {
    'public': 'ALL19_PINNED_PUBLIC_RUNTIME_ASSETS_NATIVE_SHA_VERIFIED',
    'inputs': 'EXACT_METADATA_AND_ALL16540_JPEG_MEMBER_INPUTS_RESTORED',
    'conditions': 'ALL30_FULL_AND_OOF_CACHES_ORIGINAL_EXACT_SHA_RESTORED',
    'latent': 'ALL13230_T_LATENTS_ORIGINAL_CHUNKS_AND_FILE_SHA_RESTORED',
    'evaluation': 'ALL25_ORIGINAL_PUBLIC_EVALUATOR_ASSETS_EXACT_BYTES_RESTORED',
}
HELPERS = {
    'public': 'restore_BrainHIVE_public_runtime_CPU_20261022_v1.py',
    'inputs': 'restore_BrainHIVE_input_containers_CPU_20261022_v1.py',
    'conditions': 'restore_BrainHIVE_conditions_GPU_20261022_v1.py',
    'latent': 'restore_BrainHIVE_T_latents_GPU_20261022_v1.py',
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path, read_pins):
    path = Path(path)
    raw = path.read_bytes()
    read_pins[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def exact_records(expected, records, count, path_key='path'):
    """Reject truncated proofs, duplicate aliases, added files and changed SHAs."""
    assert len(expected) == len(records) == count
    assert len({r[path_key] for r in records}) == count
    observed = {r[path_key]: r['sha256'] for r in records}
    assert observed == expected


def expected_core(read_pins):
    manifest = load(ROOT / 'configs/BrainHIVE_runtime_restore_manifest_20261022_v1.json', read_pins)
    metadata = load(ROOT / 'audit/BrainHIVE_storage_metadata_exact_archive_v14.json', read_pins)
    containers = load(ROOT / 'audit/runtime_input_containers_restored_20261022_v1.json', read_pins)
    contract = load(WORK / 'data/component_input_contract_v1.json', read_pins)
    latent = load(WORK / 'audit/component_T_latent_bank_v2.json', read_pins)
    public = {r['path']: r['sha256'] for r in manifest['assets']}
    inputs = {r['path']: r['sha256'] for r in metadata['files']}
    assert len(inputs) == 4
    inputs[containers['original_manifest_path']] = containers['container_sha256']
    conditions = {r['condition_cache_path']: r['condition_cache_sha256']
        for subject in contract['subjects'] for r in subject['sources']}
    assert len(contract['subjects']) == 10
    assert all({r['role'] for r in subject['sources']} == {'full', 'oof0', 'oof1'}
        for subject in contract['subjects'])
    return dict(public=public, inputs=inputs, conditions=conditions,
        latent={latent['path']: latent['sha256']})


def stage_files(stage, receipt):
    assert receipt['status'] == STAGE_STATUS[stage]
    if stage in ('public', 'conditions'):
        return receipt['files']
    if stage == 'inputs':
        assert receipt['images_member_SHA_verified'] == 16540
        return receipt['metadata'] + [dict(path=receipt['original_manifest_path'], sha256=receipt['container_sha256'])]
    if stage == 'latent':
        assert receipt['T_images'] == 13230 and receipt['historical_receipt_unchanged'] is True
        assert receipt['training_updates'] == receipt['UNet_forwards'] == receipt['images_generated'] == 0
        return [dict(path=receipt['path'], sha256=receipt['sha256'])]
    return receipt['assets']


def validate_copy(expected, records):
    exact_records(expected, records, len(expected), 'original_path')
    verified = []
    for row in records:
        original = Path(row['original_path'])
        ram = Path(row['RAM_path'])
        durable = Path(row['durable_path'])
        assert ram.is_absolute() and '..' not in ram.parts and ram.is_relative_to('/dev/shm')
        assert original.resolve(strict=True) == ram.resolve(strict=True)
        assert ram.resolve(strict=True).is_relative_to('/dev/shm')
        assert durable == DURABLE / 'rootfs' / ram.relative_to('/')
        assert not durable.is_symlink() and durable.resolve(strict=True).is_relative_to(DURABLE.resolve(strict=True))
        assert ram.is_file() and durable.is_file()
        assert ram.stat().st_size == durable.stat().st_size == row['bytes']
        assert sha(original) == sha(ram) == sha(durable) == expected[str(original)]
        verified.append(dict(original_path=str(original), RAM_path=str(ram), durable_path=str(durable),
            bytes=row['bytes'], sha256=row['sha256']))
    return verified


def verify_stage(stage, expected, read_pins):
    copy_path = AUDIT / (stage + '_durable.json')
    copied = load(copy_path, read_pins)
    assert copied['status'] == 'EXACT_RUNTIME_BYTES_RETAINED_ON_EXPANDED_PRIVATE_DISK'
    assert copied['stage'] == stage
    exact_records(expected, copied['files'], COUNTS[stage], 'original_path')
    if stage == 'evaluation':
        driver_path = AUDIT / 'evaluation_driver.json'
        driver = load(driver_path, read_pins)
        assert driver['status'] == 'ALL25_FROZEN_EVALUATOR_ASSETS_HAVE_VERIFIED_DURABLE_COPIES'
        assert driver['assets'] == 25 and driver['GPU_load_verified'] is False
        stage_path = Path(driver['asset_receipt']['path'])
        assert stage_path.parent == ROOT / 'audit'
        assert sha(stage_path) == driver['asset_receipt']['sha256']
        assert driver['durable_receipt'] == dict(path=str(copy_path), sha256=sha(copy_path))
        assert driver['source_sha256'] == sha(ROOT / 'restore_BrainHIVE_new_instance_evaluation_CPU_20261022_v2.py')
        assert all(sha(p) == digest for p, digest in driver['helper_sha256'].items())
    else:
        stage_path = AUDIT / (stage + '.json')
        driver_path = AUDIT / (stage + '_driver.json')
        driver = load(driver_path, read_pins)
        assert driver['source_sha256'] == sha(ROOT / 'restore_BrainHIVE_new_instance_20261022_v2.py')
        assert driver['original_helper_sha256'] == sha(ROOT / HELPERS[stage])
        assert driver['stage_receipt_sha256'] == sha(stage_path)
        assert driver['durable_receipt_sha256'] == sha(copy_path)
        assert driver['durable_files'] == COUNTS[stage]
        assert driver['fixed_scientific_sources_unchanged'] is True and driver['cleanup_performed'] is False
    result = load(stage_path, read_pins)
    exact_records(expected, stage_files(stage, result), COUNTS[stage])
    assert copied['total_payload_bytes'] == sum(r['bytes'] for r in copied['files'])
    return dict(stage=stage, files=validate_copy(expected, copied['files']),
        stage_receipt=dict(path=str(stage_path), sha256=sha(stage_path)),
        durable_receipt=dict(path=str(copy_path), sha256=sha(copy_path)),
        driver_receipt=dict(path=str(driver_path), sha256=sha(driver_path)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', action='store_true', help='Also require all 25 evaluator assets and copies')
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    assert ROOT == SERVER_ROOT and os.name == 'posix' and 1 <= args.attempt <= 3
    output = AUDIT / f'exact_coverage_closure_{"core_and_evaluation" if args.evaluation else "core"}_attempt{args.attempt}.json'
    assert not output.exists() and not output.with_name(output.name + '.pending').exists()
    started = time.monotonic(); read_pins = {}
    assert DURABLE.resolve().is_relative_to(DATA.resolve())
    assert DURABLE.stat().st_dev == DATA.stat().st_dev != Path('/dev/shm').stat().st_dev
    expected = expected_core(read_pins)
    recipe = load(ROOT / 'configs/functional_profiles_20261022_v1/O.json', read_pins)
    assert len(recipe['input_sha256']) == 37
    assert all(sha(p) == digest for p, digest in recipe['input_sha256'].items())
    assert all(sha(ROOT / p) == digest for p, digest in recipe['source_sha256'].items())
    if args.evaluation:
        # Derive original evaluator identities independently of the new copy list.
        import restore_BrainHIVE_evaluation_runtime_CPU_20261022_v1 as recovery
        manifests = load(ROOT / 'audit/seed_hf_selected_assets.json', read_pins)
        archive = ROOT / 'seed_small_configs.zip'; read_pins[str(archive)] = sha(archive)
        small, large = recovery.inspect_small(manifests, archive)
        direct = load(ROOT / 'audit/seed_direct_assets_receipt.json', read_pins)
        assert direct['status'] == 'OFFICIAL_PUBLIC_EVALUATOR_ASSETS_VERIFIED' and len(direct['assets']) == 2
        expected['evaluation'] = {r['path']: r['sha256'] for r in small + large + direct['assets']}
    assert sum(len(v) for v in expected.values()) == (80 if args.evaluation else 55)
    stages = [verify_stage(stage, records, read_pins) for stage, records in expected.items()]
    assert all(sha(path) == digest for path, digest in read_pins.items())
    value = dict(status='EXACT_ORIGINAL_IDENTITY_COVERAGE_RAM_AND_DURABLE_RUNTIME_CLOSURE_VERIFIED',
        UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), source_sha256=sha(__file__),
        original_identity_inputs_sha256=read_pins, core_original_records=55,
        evaluator_original_records=25 if args.evaluation else 0,
        direct_profile_input_SHA_count=37, source_pins_unchanged=True,
        stages=stages, seconds=time.monotonic()-started,
        cleanup_performed=False, GPU_used=False, score_tables_read=0,
        formal_training_admitted=False, GPU_evaluation_probe_verified=False,
        full_lifecycle_closure_verified=False)
    raw = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    assert len(raw) <= 256 << 10
    temporary = output.with_name(output.name + '.pending')
    with temporary.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(output)
    print(json.dumps(dict(status=value['status'], receipt=str(output), original_records=55+(25 if args.evaluation else 0))), flush=True)


if __name__ == '__main__':
    main()
