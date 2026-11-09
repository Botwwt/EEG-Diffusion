"""Fetch public evaluator files from their official HTTPS hosts into server RAM."""
from pathlib import Path
import hashlib
import json
import os
import time
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parent
OUT = Path('/dev/shm/eeg_sota_autoresearch_20261018_v1/semantic_evaluation')
ASSETS = [
    ('grounding_dino_swin-l_pretrain_all-56d69e78.pth',
     'https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-l_pretrain_all/grounding_dino_swin-l_pretrain_all-56d69e78.pth', '56d69e78'),
    ('efficientnet_b1-c27df63c.pth',
     'https://download.pytorch.org/models/efficientnet_b1-c27df63c.pth', 'c27df63c'),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    records = []
    for name, url, expected_prefix in ASSETS:
        target = OUT / name
        partial = OUT / (name + '.download')
        if not target.exists():
            for attempt in range(4):
                assert time.monotonic() - started < 1800
                offset = partial.stat().st_size if partial.exists() else 0
                headers = {'Accept-Encoding': 'identity', 'User-Agent': 'EEG-evaluator-acquisition/1.0'}
                if offset:
                    headers['Range'] = f'bytes={offset}-'
                try:
                    request = urllib.request.Request(url, headers=headers)
                    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=25) as response:
                        if offset:
                            assert response.status == 206
                            assert response.headers['Content-Range'].startswith(f'bytes {offset}-')
                        expected_bytes = offset + int(response.headers['Content-Length'])
                        with partial.open('ab' if offset else 'xb') as file:
                            while block := response.read(4 * 1024**2):
                                file.write(block)
                                (OUT / 'direct_progress.json').write_text(json.dumps(dict(
                                    asset=name, bytes=file.tell(), expected_bytes=expected_bytes,
                                    seconds=time.monotonic()-started, pid=os.getpid())) + '\n')
                                assert time.monotonic() - started < 1800
                    assert partial.stat().st_size == expected_bytes
                    digest = hashlib.file_digest(partial.open('rb'), 'sha256').hexdigest()
                    assert digest.startswith(expected_prefix), (name, digest)
                    os.replace(partial, target)
                    break
                except Exception:
                    if attempt == 3:
                        raise
            assert target.exists()
        digest = hashlib.file_digest(target.open('rb'), 'sha256').hexdigest()
        assert digest.startswith(expected_prefix)
        records.append(dict(path=str(target), url=url, bytes=target.stat().st_size,
                            sha256=digest, official_filename_sha256_prefix=expected_prefix,
                            full_sha256_independently_published=False))
        print(json.dumps(records[-1]), flush=True)
    (ROOT/'audit/seed_direct_assets_receipt.json').write_text(json.dumps(dict(
        status='OFFICIAL_PUBLIC_EVALUATOR_ASSETS_VERIFIED', assets=records,
        seconds=time.monotonic()-started, local_large_files_created=0), indent=2)+'\n')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        (ROOT/'audit/seed_direct_assets_failure.json').write_text(traceback.format_exc())
        raise
