"""Reuse byte-identical thumbnails while running the unchanged offline builder.

The measured plan and storage gate precede construction. Every cached thumbnail
must match a fresh canonical encoding of its hash-verified full JPEG. The save
interception only replaces that thumbnail's storage with a hard link; the original
builder still checks all original images and creates its usual complete materials.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import math
import os
import runpy
import shutil
import sys
import time
from PIL import Image, __version__ as pillow_version

ROOT = Path(__file__).resolve().parent
DATA = Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')
ORIGINAL_BUILDER_SHA = 'f046276975824e02c0b592199ebd27fa31babe895097f72166f71c5a96ca1523'


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def thumbnail(path):
    with Image.open(path) as image:
        image.load()
        buffer = io.BytesIO()
        image.convert('RGB').resize((256, 256), Image.Resampling.LANCZOS).save(
            buffer, format='JPEG', quality=86, subsampling=0)
        return buffer.getvalue()


def prepare_plan(ready, cache_roots, output):
    started = time.monotonic()
    output = Path(output).resolve()
    assert output.is_relative_to(DATA/'galleries') and not output.exists()
    builder = ROOT/'build_offline_gallery.py'
    assert sha(builder) == ORIGINAL_BUILDER_SHA
    caches = [Path(p).resolve() for p in cache_roots]
    assert caches and len(caches) == len(set(caches))
    for cache in caches:
        assert cache.is_relative_to(DATA/'galleries') and cache != DATA/'galleries'
        complete = load(cache/'completion.json')
        assert complete['status'] == 'COMPLETED_FULL_OFFLINE_GALLERY'
        assert complete['queries'] == 16600 and complete['missing_images'] == complete['corrupt_images'] == 0
    assets = {}
    source_receipts = []
    image_totals, feature_totals = [], []
    for source in ready:
        rows = [json.loads(line) for line in Path(source['manifest']).read_text().splitlines()]
        with Path(source['metrics']).open(newline='') as file:
            scores = {s['query_id']: s for s in csv.DictReader(file)}
        assert len(rows) == len(scores) == 16600
        image_totals.append(sum(Path(r['generated_path']).stat().st_size for r in rows))
        features = DATA/'evaluation/V'/source['name']/'features'
        feature_totals.append(sum(p.stat().st_size for p in features.rglob('*.pt')))
        source_receipts.append({k: source[k] for k in ['name', 'manifest', 'manifest_sha256', 'metrics', 'metrics_sha256']})
        for row in rows:
            score = scores[row['query_id']]
            for path, expected in [(score['score_target_path'], row['target_image_sha256']),
                                   (row['generated_path'], row['generated_sha256'])]:
                key = hashlib.sha256((str(path)+'|'+expected).encode()).hexdigest()[:32]
                if key in assets:
                    assert assets[key]['source'] == path and assets[key]['source_sha256'] == expected
                    continue
                assert sha(path) == expected
                encoded = thumbnail(path)
                digest = hashlib.sha256(encoded).hexdigest()
                cached = None
                for cache in caches:
                    thumb = cache/'blinded/thumbs'/f'{key}.jpg'
                    full = cache/'blinded/full'/f'{key}.jpg'
                    if thumb.is_file() and full.is_file():
                        assert thumb.resolve().is_relative_to(DATA/'galleries')
                        assert full.resolve().is_relative_to(DATA/'galleries')
                        assert sha(full) == expected
                        assert thumb.stat().st_size == len(encoded) and sha(thumb) == digest
                        assert thumb.stat().st_dev == output.parent.stat().st_dev
                        cached = str(thumb)
                        break
                assets[key] = dict(source=path, source_sha256=expected,
                                   thumbnail_sha256=digest, thumbnail_bytes=len(encoded), cache_thumb=cached)
    fresh = sum(a['thumbnail_bytes'] for a in assets.values() if a['cache_thumb'] is None)
    # Preserve a 1GiB reserve and allow the already admitted full Frozen lifecycle.
    frozen = DATA/'evaluation/V/IDEA1_Frozen_correct_R4'
    pending_images = not (DATA/'images/IDEA1_interface_v1/V/IDEA1_Frozen_correct_R4/completion.json').exists()
    pending_features = not (frozen/'features/completion.json').exists()
    pending = math.ceil(1.1*((max(image_totals) if pending_images else 0)+(max(feature_totals) if pending_features else 0)))
    estimate = 2*fresh + 256*1024**2
    free = shutil.disk_usage(output.parent).free
    plan = dict(status='MEASURED_CANONICAL_THUMBNAIL_PLAN', output=str(output),
                source_sha256=sha(__file__), original_builder_sha256=sha(builder),
                pillow_version=pillow_version, sources=source_receipts,
                caches=[dict(path=str(c), completion_sha256=sha(c/'completion.json')) for c in caches],
                assets=assets, unique_assets=len(assets),
                cached_assets=sum(a['cache_thumb'] is not None for a in assets.values()),
                new_thumbnail_bytes=fresh, estimated_new_bytes_with_margin=estimate,
                ongoing_Frozen_lifecycle_reserve_bytes=pending, minimum_free_reserve_bytes=1024**3,
                free_bytes_at_gate=free, storage_gate_passed=free>=estimate+pending+1024**3,
                GPU_used=False, seconds=time.monotonic()-started)
    path = output.parent/(output.name+'.thumbnail-plan.json')
    with path.open('x') as file:
        file.write(json.dumps(plan, separators=(',', ':'))+'\n')
    assert plan['storage_gate_passed'], ('Measured export plus ongoing Frozen and 1GiB reserve do not fit',
                                         free, estimate, pending, str(path))
    return path, {k: v for k, v in plan.items() if k != 'assets'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--thumbnail-plan', required=True)
    args, builder_args = parser.parse_known_args()
    plan_path = Path(args.thumbnail_plan).resolve()
    assert plan_path.is_relative_to(DATA/'galleries')
    plan = load(plan_path)
    assert plan['storage_gate_passed'] and plan['source_sha256'] == sha(__file__)
    assert plan['pillow_version'] == pillow_version
    output = Path(builder_args[builder_args.index('--output')+1]).resolve()
    assert str(output) == plan['output'] and not output.exists()
    builder = ROOT/'build_offline_gallery.py'
    assert sha(builder) == plan['original_builder_sha256'] == ORIGINAL_BUILDER_SHA
    free = shutil.disk_usage(output.parent).free
    assert free >= plan['estimated_new_bytes_with_margin']+plan['ongoing_Frozen_lifecycle_reserve_bytes']+1024**3
    original_save = Image.Image.save
    touched, links = set(), []
    started = time.monotonic()

    def save(image, fp, format=None, **params):
        path = Path(fp) if isinstance(fp, (str, Path)) else None
        if path is None or path.parent != output/'blinded/thumbs':
            return original_save(image, fp, format=format, **params)
        assert format is None and params == dict(quality=86, subsampling=0)
        key = path.stem
        assert key in plan['assets'] and key not in touched and not path.exists()
        entry = plan['assets'][key]
        buffer = io.BytesIO()
        original_save(image, buffer, format='JPEG', **params)
        encoded = buffer.getvalue()
        assert len(encoded) == entry['thumbnail_bytes']
        assert hashlib.sha256(encoded).hexdigest() == entry['thumbnail_sha256']
        if entry['cache_thumb'] is not None:
            cached = Path(entry['cache_thumb'])
            assert cached.stat().st_size == len(encoded) and sha(cached) == entry['thumbnail_sha256']
            os.link(cached, path)
            assert path.stat().st_ino == cached.stat().st_ino and path.stat().st_dev == cached.stat().st_dev
            links.append(key)
        else:
            with path.open('xb') as file:
                file.write(encoded)
        touched.add(key)

    Image.Image.save = save
    previous = sys.argv
    try:
        sys.argv = [str(builder), *builder_args]
        runpy.run_path(str(builder), run_name='__main__')
    finally:
        Image.Image.save = original_save
        sys.argv = previous
    assert touched == plan['assets'].keys()
    complete = load(output/'completion.json')
    assert complete['assets'] == len(touched) and complete['builder_sha256'] == ORIGINAL_BUILDER_SHA
    result = dict(status='PASSED_CANONICAL_THUMBNAIL_BYTES_AND_HARDLINK_REUSE', GPU_used=False,
                  assets=len(touched), cached_thumbnail_hardlinks=len(links),
                  newly_written_thumbnail_assets=len(touched)-len(links),
                  new_thumbnail_bytes=plan['new_thumbnail_bytes'],
                  plan=str(plan_path), plan_sha256=sha(plan_path), source_sha256=sha(__file__),
                  original_builder_sha256=ORIGINAL_BUILDER_SHA,
                  seconds=time.monotonic()-started, human_ratings_performed=False,
                  scientific_goal_complete=False)
    (output/'thumbnail_hardlink_receipt.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
