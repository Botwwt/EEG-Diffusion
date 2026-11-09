"""Joint remaining-life accounting for two fixed experiments, never a disk promise.

Existing main-arm forecasts remain protected. A small physical metadata reserve
is explicitly separate from the forecast for future checkpoints and images.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat

ROOT = Path(__file__).resolve().parent
WORK = Path('/root/brainhive_reproduction_20261020_v1')
DATA = Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def core_sha(recipe):
    value = {k: v for k, v in recipe.items()
             if k not in {'storage_admission_receipt', 'storage_admission_sha256'}}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def size(path):
    path = Path(path)
    return sum(p.stat().st_size for p in path.rglob('*')
               if p.is_file() and not p.is_symlink()) if path.exists() else 0


def checked_root(path):
    p = Path(path)
    assert p.is_absolute() and not p.is_symlink()
    assert p.resolve().is_relative_to(WORK) or p.resolve().is_relative_to(DATA)
    assert p.resolve() not in {WORK, DATA}
    return p


def protected_main():
    from brainhive_lifecycle_storage_v1 import verify_remaining
    recipe = read(ROOT / 'configs/BrainHIVE_bounded_main_recipe_v1.json')
    spec = read(recipe['storage_lifecycle_receipt'])
    report = verify_remaining(recipe)
    required = {}
    for root, need in zip(spec['allowed_private_roots'], report['remaining_required']):
        dev = Path(root).stat().st_dev
        required[dev] = required.get(dev, 0) + need
    return required, report


def forecast(recipes):
    """Fresh joint bound; does not change recipes or count RAM as durable disk."""
    required, main = protected_main()
    by_device = {}
    for p in [WORK, DATA]:
        dev = p.stat().st_dev
        by_device[dev] = dict(path=str(p), free=shutil.disk_usage(p).free,
                              protected_main_bytes=required.get(dev, 0), new_bytes=0)
    detail = []
    for recipe in recipes:
        candidate = recipe['candidate']
        assert candidate in {'brainkan_full_v1', 'response_rank_v1'}
        assert recipe['formal_V_queries_per_arm'] == 16600
        assert recipe['reserved_bytes_per_image'] == 330000
        roots = [checked_root(p) for p in recipe['allowed_private_roots']]
        segments = recipe['image_segments']
        assert segments[0]['start'] == 0 and segments[-1]['stop'] == 16600
        assert all(a['stop'] == b['start'] for a, b in zip(segments, segments[1:]))
        manifest = WORK / 'images' / recipe['run_name'] / 'inference_manifest.jsonl'
        rows = [json.loads(line) for line in manifest.read_text().splitlines()] if manifest.exists() else []
        done = {r['global_query_index'] for r in rows}
        assert len(done) == len(rows) and done <= set(range(16600))
        image_bytes = 0
        for segment in segments:
            root = Path(segment['root'])
            assert root in roots and segment['stop'] > segment['start']
            # An ancestor must exist before admission; no storage on C/D/public.
            ancestor = root
            while not ancestor.exists():
                ancestor = ancestor.parent
            dev = ancestor.stat().st_dev
            assert dev in by_device
            count = sum(i not in done for i in range(segment['start'], segment['stop']))
            needed = count * recipe['reserved_bytes_per_image']
            by_device[dev]['new_bytes'] += needed
            image_bytes += needed
        training = max(0, recipe['training_artifact_budget_bytes'] - size(recipe['output_directory']))
        features = max(0, recipe['feature_budget_bytes'] - size(WORK / 'evaluation' / recipe['run_name'] / 'features'))
        metadata_written = size(WORK / 'images' / recipe['run_name'])
        metadata_written += size(WORK / 'evaluation' / recipe['run_name'] / 'scores')
        metadata_written += size(WORK / 'evaluation' / recipe['run_name'] / 'low_level')
        metadata = max(0, recipe['metadata_budget_bytes'] - metadata_written)
        by_device[WORK.stat().st_dev]['new_bytes'] += training + features + metadata
        detail.append(dict(candidate=candidate, images_remaining=16600-len(done), image_bytes=image_bytes,
                           training_bytes=training, features_bytes=features, metadata_bytes=metadata))
    volumes = list(by_device.values())
    for v in volumes:
        v['required'] = v['protected_main_bytes'] + v['new_bytes']
        v['deficit'] = max(0, v['required'] - v['free'])
    return dict(funded=all(v['deficit'] == 0 for v in volumes), volumes=volumes,
                candidates=detail, protected_main=main,
                total_deficit=sum(v['deficit'] for v in volumes),
                forecast_not_physical_full_preallocation=True)


def verify_candidate_remaining(recipe, candidate):
    assert recipe['candidate'] == candidate
    for filename, digest in recipe['source_sha256'].items():
        assert sha(ROOT / filename) == digest, filename
    receipt_path = Path(recipe['storage_admission_receipt'])
    assert sha(receipt_path) == recipe['storage_admission_sha256']
    receipt = read(receipt_path)
    assert receipt['status'] == 'FULL_TRAIN_GENERATE_SCORE_ADMITTED'
    assert receipt['candidate'] == candidate and receipt['recipe_core_sha256'] == core_sha(recipe)
    assert receipt['output_directory'] == recipe['output_directory']
    assert set(receipt['scope']) >= {'training', 'actual_images', 'per_query_evaluation'}
    assert receipt['max_training_artifact_bytes'] == recipe['training_artifact_budget_bytes']
    assert receipt['reservation_paths']
    for item in receipt['reservation_paths']:
        p = checked_root(item['path'])
        assert p.is_file() and stat.S_IMODE(p.stat().st_mode) == 0o600
        assert p.stat().st_blocks * 512 >= item['minimum_allocated_bytes'] >= 16 << 20
    plan_path = Path(receipt['group_plan'])
    assert sha(plan_path) == receipt['group_plan_sha256']
    plan = read(plan_path)
    siblings = []
    assert len(plan['recipes']) == 2
    assert len({item['path'] for item in plan['recipes']}) == 2
    for item in plan['recipes']:
        sibling = read(item['path'])
        assert core_sha(sibling) == item['core_sha256']
        siblings.append(sibling)
    assert {s['candidate'] for s in siblings} == {'brainkan_full_v1', 'response_rank_v1'}
    own = [s for s in siblings if s['candidate'] == candidate]
    assert len(own) == 1 and core_sha(own[0]) == core_sha(recipe)
    report = forecast(siblings)
    assert report['funded'], 'Joint complete experiment storage no longer funded: ' + json.dumps(report['volumes'])
    return report
