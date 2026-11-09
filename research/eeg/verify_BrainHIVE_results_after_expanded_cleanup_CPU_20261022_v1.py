"""Recheck result pixels and final weights after expanded disposable cleanup."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
WORK = Path('/root/brainhive_reproduction_20261020_v1')
OUT = ROOT / 'audit/results_after_expanded_cleanup_actual_20261022_v1.json'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    assert os_name_ok() and not OUT.exists()
    started = time.monotonic()
    previous_path = ROOT / 'audit/retained_results_verification_20261022_v1.json'
    previous = json.loads(previous_path.read_text())
    counts = {}; seen = set(); evidence = hashlib.sha256()
    for manifest in sorted((WORK / 'images').glob('*/manifest.jsonl')):
        count = 0
        for line in manifest.open():
            row = json.loads(line); path = row['generated_path']
            assert sha(path) == row['generated_sha256'], path
            count += 1; seen.add(path)
            evidence.update(json.dumps([path, row['generated_sha256']], separators=(',', ':')).encode() + b'\n')
        counts[manifest.parent.name] = count
        print(json.dumps(dict(verified_arm=manifest.parent.name, count=count)), flush=True)
    assert counts == previous['image_counts'] and len(seen) == 55832
    weights = previous['durable_final_checkpoints_SHA']
    assert len(weights) == 52
    for path, expected in weights.items():
        assert sha(path) == expected, path
    assert sha(previous['archive_path']) == previous['archive_sha256']
    recipe = json.loads((ROOT / 'configs/functional_profiles_20261022_v1/O.json').read_text())
    for name, digest in recipe['source_sha256'].items():
        assert sha(ROOT / name) == digest, name
    result_digest = sha(ROOT / 'results.tsv')
    assert result_digest == '2ee905cc31949671c3068f04c9dbd05046ed4ae9888b05c9ca7cb08d781fb010'
    value = dict(status='ALL55832_RESULT_IMAGES_AND52_FINAL_MODELS_RECHECKED_EXACT_SHA',
        UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), source_sha256=sha(__file__),
        previous_result_receipt_sha256=sha(previous_path), image_counts=counts,
        result_images_SHA_verified=len(seen), ordered_path_SHA_evidence_digest=evidence.hexdigest(),
        final_checkpoints_SHA_verified=52, registered_current_EEG_models=40,
        current_components=2, extra_historical_EEG_models=10,
        result_metadata_backup_SHA_unchanged=True, results_tsv_sha256=result_digest,
        pinned_scientific_sources_unchanged=True, GPU_calls=0,
        score_values_read=False, new_scientific_results=False, seconds=time.monotonic() - started)
    with OUT.open('x') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')
    print(json.dumps(value), flush=True)


def os_name_ok():
    import os
    return os.name == 'posix' and ROOT == Path('/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1')


if __name__ == '__main__':
    main()
