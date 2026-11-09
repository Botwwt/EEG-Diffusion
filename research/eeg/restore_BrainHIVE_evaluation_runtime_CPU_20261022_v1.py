"""Restore the original public evaluator bytes; never rewrite prior receipts.

Weights remain in server RAM. Existing EEG, images, features and scores are read
only. This file does not load models, fit anything, or admit formal experiments.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.parse
import urllib.request
import zipfile

import restore_BrainHIVE_public_runtime_CPU_20261022_v1 as transfer

ROOT = Path(__file__).resolve().parent
BASE = Path('/dev/shm/eeg_sota_autoresearch_20261018_v1/semantic_evaluation')
GROUPS = {'microsoft/git-large-coco': 'git_large_coco',
          'sentence-transformers/all-MiniLM-L6-v2': 'sentence_minilm',
          'google-bert/bert-base-uncased': 'bert_base'}
REVISIONS = {'microsoft/git-large-coco': 'b644b0b41274649f87de8846fa15f9a22dad7583',
             'sentence-transformers/all-MiniLM-L6-v2': '1110a243fdf4706b3f48f1d95db1a4f5529b4d41',
             'google-bert/bert-base-uncased': '86b5e0934494bd15c9632b12f734a8a67f723594'}
DIRECT_HOSTS = {'download.openmmlab.com', 'download.pytorch.org'}
DINO = Path('/root/mced_brain_dinov3_s_v1/dinov3_vits16_pretrain_lvd1689m-08c60483.pth')
DINO_SHA = '08c60483bc63c04f533611e34bf70b120eedb7240f469bc16e9e20bf344b941d'


def blob_oid(raw):
    return hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()


def inspect_small(manifests, archive):
    """Validate the retained config archive against original Git blob identities."""
    assert {m['repository']: m['revision'] for m in manifests} == REVISIONS
    small, large = [], []
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist()) == len(set(z.namelist()))
        for manifest in manifests:
            for item in manifest['assets']:
                relative = Path(GROUPS[manifest['repository']]) / item['path']
                assert not relative.is_absolute() and '..' not in relative.parts
                common = dict(path=str(BASE / relative), bytes=item['size'])
                if 'lfs' in item:
                    assert item['size'] == item['lfs']['size']
                    large.append(common | dict(repo=manifest['repository'], revision=manifest['revision'],
                        source_path=item['path'], sha256=item['lfs']['oid']))
                else:
                    assert item['size'] <= 1 << 20
                    raw = z.read(relative.as_posix())
                    assert len(raw) == item['size'] and blob_oid(raw) == item['oid'], str(relative)
                    small.append(common | dict(raw=raw, git_blob_oid=item['oid'],
                        sha256=hashlib.sha256(raw).hexdigest()))
        assert set(z.namelist()) == {str(Path(x['path']).relative_to(BASE)).replace('\\', '/') for x in small}
    assert len(small) == 20 and len(large) == 3
    return small, large


def direct_range(asset, lo, deadline):
    hi = min(lo + transfer.CHUNK, asset['bytes']) - 1
    url = asset['url']
    parsed = urllib.parse.urlsplit(url)
    assert parsed.scheme == 'https' and parsed.hostname in DIRECT_HOSTS
    for attempt in range(3):
        assert time.monotonic() < deadline
        try:
            request = urllib.request.Request(url + '?restore_20261022=' + str(lo),
                headers={'Range': f'bytes={lo}-{hi}', 'Accept-Encoding': 'identity'})
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=40) as response:
                assert response.status == 206
                assert response.headers['Content-Range'] == f"bytes {lo}-{hi}/{asset['bytes']}"
                assert urllib.parse.urlsplit(response.url).hostname in DIRECT_HOSTS
                data = response.read(transfer.CHUNK + 1)
            assert len(data) == hi - lo + 1
            return lo, data
        except Exception:
            if attempt == 2:
                raise


def restore_small(asset):
    path = transfer.destination(asset)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists():
        assert path.stat().st_size == asset['bytes'] and transfer.sha(path) == asset['sha256']
        reused = True
    else:
        with path.open('xb') as stream:
            stream.write(asset['raw']); stream.flush(); os.fsync(stream.fileno())
        reused = False
    assert transfer.sha(path) == asset['sha256']
    return {k: v for k, v in asset.items() if k != 'raw'} | dict(reused=reused)


def bound(path):
    return dict(path=str(path), sha256=transfer.sha(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    assert os.name == 'posix' and ROOT == Path('/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1')
    assert 1 <= args.workers <= 64 and 1 <= args.attempt <= 3
    started = time.monotonic(); deadline = started + 3600
    output = ROOT / 'audit' / f'runtime_evaluation_public_restore_20261022_v1_attempt{args.attempt}.json'
    assert not output.exists(), 'Keep every prior attempt and its outcome'
    dependencies = [ROOT / 'audit/seed_hf_selected_assets.json', ROOT / 'seed_small_configs.zip',
        ROOT / 'audit/seed_hf_acquisition_completion.json', ROOT / 'audit/seed_direct_assets_receipt.json',
        Path(transfer.__file__), ROOT / 'image_evaluation.py', ROOT / 'semantic_evaluator.py',
        ROOT / 'extract_image_features.py']
    pinned = {str(p): transfer.sha(p) for p in dependencies}
    small, large = inspect_small(json.loads(dependencies[0].read_text()), dependencies[1])
    direct = json.loads((ROOT / 'audit/seed_direct_assets_receipt.json').read_text())
    assert direct['status'] == 'OFFICIAL_PUBLIC_EVALUATOR_ASSETS_VERIFIED'
    assert len(direct['assets']) == 2
    for item in direct['assets']:
        assert Path(item['path']).parent == BASE
        assert urllib.parse.urlsplit(item['url']).hostname in DIRECT_HOSTS
    assets = small + large + direct['assets']
    assert len({a['path'] for a in assets}) == 25
    required = sum(a['bytes'] for a in assets if not Path(a['path']).is_file())
    assert shutil.disk_usage('/dev/shm').free >= required + (3 << 30)
    assert shutil.disk_usage(ROOT).free >= 16 << 20
    assert DINO.is_file() and transfer.sha(DINO) == DINO_SHA
    initial = dict(status='RESTORING_ORIGINAL_PUBLIC_EVALUATOR_ASSETS_NO_MODELS_LOADED',
        UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), pid=os.getpid(),
        source=bound(Path(__file__)), input_sha256=pinned, maximum_seconds=3600,
        maximum_durable_metadata_bytes=16 << 20, required_missing_RAM_bytes=required,
        formal_training_started=False, new_generated_images=0, score_tables_read=0)
    transfer.write(output, initial)
    transfer.START = started; transfer.WORKERS = args.workers
    transfer.PROGRESS = ROOT / 'audit/runtime_evaluation_public_restore_progress_20261022_v1.json'
    original_fetch = transfer.fetch_range

    def fetch(asset, lo):
        assert time.monotonic() < deadline
        return direct_range(asset, lo, deadline) if 'url' in asset else original_fetch(asset, lo)

    transfer.fetch_range = fetch
    try:
        records = [restore_small(a) for a in small]
        for asset in large + direct['assets']:
            assert time.monotonic() < deadline
            records.append(transfer.restore(asset))
            print(json.dumps(dict(verified_public_asset=asset['path'], bytes=asset['bytes'])), flush=True)
        assert all(transfer.sha(p) == digest for p, digest in pinned.items())
        assert len(records) == 25 and all(transfer.sha(x['path']) == x['sha256'] for x in records)
        initial.update(status='ALL25_ORIGINAL_PUBLIC_EVALUATOR_ASSETS_EXACT_BYTES_RESTORED',
            end_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), seconds=time.monotonic()-started,
            assets=records, existing_DINO=bound(DINO), original_asset_receipts_unchanged=True,
            total_RAM_payload_bytes=sum(x['bytes'] for x in records), durable_weight_bytes_created=0,
            evaluation_model_loads_verified=False, full_lifecycle_storage_admitted=False)
        transfer.write(output, initial)
        print(json.dumps(dict(status=initial['status'], receipt=str(output))), flush=True)
    except Exception as error:
        initial.update(status='EVALUATOR_ASSET_RESTORATION_FAILED_PRESERVED_PARTIALS',
            error_type=type(error).__name__, error=str(error)[:600], seconds=time.monotonic()-started)
        transfer.write(output, initial)
        raise


if __name__ == '__main__':
    main()
