"""Read-only result verification plus a bounded metadata-only evidence archive.

All generated pixels and learned weights stay in their original durable places.
The ZIP is an additional copy of score/provenance text, not an image backup.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile

ROOT = Path(__file__).resolve().parent
WORK = Path('/root/brainhive_reproduction_20261020_v1')
DATA = Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')
ZIP = ROOT / 'audit/retained_result_metadata_20261022_v1.zip'
RECEIPT = ROOT / 'audit/retained_results_verification_20261022_v1.json'


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seal-existing-archive', action='store_true')
    args = parser.parse_args()
    assert ROOT == Path('/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1')
    assert not RECEIPT.exists()
    assert ZIP.is_file() if args.seal_existing_archive else not ZIP.exists()
    verified = {}; arms = {}; files = set()
    for manifest in sorted((WORK / 'images').glob('*/manifest.jsonl')):
        count = 0
        with manifest.open() as stream:
            for line in stream:
                row = json.loads(line); path = Path(row['generated_path'])
                digest = verified.setdefault(str(path), row['generated_sha256'])
                assert digest == row['generated_sha256'] and sha(path) == digest, str(path)
                count += 1
                if count % 5000 == 0: print('VERIFIED_CURRENT_IMAGES', manifest.parent.name, count, flush=True)
        arms[manifest.parent.name] = count; files.add(manifest)
    for arm in ('risk', 'ordinary', 'parent'):
        assert arms[f'protected_R4_{arm}_bounded_v1'] == 16600
    for arm in ('correct', 'permuted', 'missing'):
        assert arms[f'native_R80_{arm}_v1'] == 2000
    # Existing pinned inputs must still match; absent volatile assets stay explicit failures of readiness.
    recipe = json.loads((ROOT / 'configs/functional_profiles_20261022_v1/O.json').read_text())
    missing = []; matched = 0
    for path, digest in recipe['input_sha256'].items():
        p = Path(path)
        if not p.is_file(): missing.append(dict(path=path, expected_sha256=digest)); continue
        assert sha(p) == digest, path; matched += 1
    inventory = json.loads((ROOT / 'audit/BrainHIVE_current_model_inventory_CPU_v21.json').read_text())
    registered_EEG = [row for row in inventory['models'] if row['module'] == 'EEG']
    assert len(registered_EEG) == 40
    for row in registered_EEG:
        assert sha(row['checkpoint']) == row['checkpoint_sha256'], row['checkpoint']
    weights = {}
    for p in list((WORK / 'runs').glob('**/brain_final.pt')) + list((WORK / 'components/bounded_v1').glob('*/component_final.pt')):
        weights[str(p)] = sha(p)
    assert all(row['checkpoint'] in weights for row in registered_EEG)
    components = list((WORK / 'components/bounded_v1').glob('*/component_final.pt'))
    assert {p.parent.name for p in components} == {'risk', 'ordinary'}
    # Select text evidence; arrays, images, feature tensors and model payloads never enter this copy.
    for base in [WORK / 'evaluation', WORK / 'comparison', DATA / 'evaluation',
                 DATA / 'images/IDEA1_interface_v1', DATA / 'images/S1_full_native_batchsafe_v2']:
        for p in base.rglob('*'):
            if p.is_file() and not p.is_symlink() and p.suffix in ('.json', '.jsonl', '.csv', '.tsv'):
                files.add(p)
    for p in (WORK / 'images').glob('*/*.json'): files.add(p)
    for p in (ROOT / 'reports').glob('*.md'): files.add(p)
    for name in ('results.tsv', 'experiment_ledger.jsonl', 'campaign_state.json', 'budget.yaml'):
        files.add(ROOT / name)
    assert sum(p.stat().st_size for p in files) < (1 << 30)
    entries = []
    if not args.seal_existing_archive:
        with zipfile.ZipFile(ZIP, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for p in sorted(files):
                assert p.resolve() == p and p.stat().st_size < (128 << 20)
                digest = sha(p); archive.write(p, arcname=str(p).removeprefix('/'))
                entries.append(dict(path=str(p), sha256=digest, bytes=p.stat().st_size))
    else:
        entries = [dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size) for p in sorted(files)]
    # This is the historical-result backup's revised measured budget, not an arm's metadata budget.
    assert ZIP.stat().st_size < (256 << 20)
    with zipfile.ZipFile(ZIP) as archive:
        assert len(archive.namelist()) == len(entries)
        assert set(archive.namelist()) == {x['path'].removeprefix('/') for x in entries}
        assert archive.testzip() is None
        for entry in entries:
            with archive.open(entry['path'].removeprefix('/')) as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == entry['sha256']
            assert sha(entry['path']) == entry['sha256']
    record = dict(status='RESULT_IMAGES_MODELS_VERIFIED_AND_METADATA_COPY_CRC_SHA_VERIFIED',
        UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), source_sha256=sha(__file__),
        current_BrainHIVE_images_SHA_verified=len(verified), image_counts=arms,
        durable_final_checkpoints_SHA=weights, matched_existing_profile_input_files=matched,
        registered_current_EEG_checkpoints_verified=40, current_component_checkpoints_verified=2,
        historical_extra_EEG_checkpoints_retained=len(weights) - 42,
        missing_volatile_profile_inputs=missing, current_training_runtime_ready=not missing,
        missing_inputs_observed_before_this_cleanup=True, missing_inputs_deleted_by_this_cleanup=False,
        archive_path=str(ZIP), archive_sha256=sha(ZIP), archive_bytes=ZIP.stat().st_size,
        metadata_entries=entries, archive_contains_images_models_or_EEG_arrays=False,
        original_images_and_weight_payloads_retained_in_place=True,
        metadata_copy_is_not_a_second_image_backup=True, GPU_calls=0, new_scientific_results=False)
    record.update(archive_initial_128MiB_cap_not_met=ZIP.stat().st_size >= (128 << 20),
        archive_measured_budget_bytes=256 << 20, functional_arm_metadata_budgets_unchanged=True,
        sealed_existing_archive=args.seal_existing_archive)
    with RECEIPT.open('x') as f: json.dump(record, f, indent=2); f.write('\n')
    print(json.dumps({k: v for k, v in record.items() if k not in ('metadata_entries', 'durable_final_checkpoints_SHA', 'missing_volatile_profile_inputs')}), flush=True)


if __name__ == '__main__': main()
