"""Restore 19 pinned public runtime files to volatile server RAM only.

No EEG downloads, fits, metrics, GPU calls, or historical receipt changes.
The private T latent must be restored separately and is never fetched online.
"""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import threading
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'configs/BrainHIVE_runtime_restore_manifest_20261022_v1.json'
OUT = ROOT / 'audit/runtime_public_restore_20261022_v1.json'
PROGRESS = ROOT / 'audit/runtime_public_restore_progress_20261022_v1.json'
CHUNK = 8 << 20
START = time.monotonic()
CDN_URLS = {}
CDN_LOCK = threading.Lock()
WORKERS = 16


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    temporary = path.with_name(path.name + '.pending')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, url):
        return None


def url(asset, offset=None):
    value = ('https://huggingface.co/' + asset['repo'] + '/resolve/'
             + asset['revision'] + '/' + asset['source_path'])
    return value + '?restore_20261022=' + str(offset) if offset is not None else value


def proxy_opener(no_redirect=False):
    handlers = [urllib.request.ProxyHandler({'https': 'http://127.0.0.1:33344'})]
    if no_redirect:
        handlers.append(NoRedirect())
    return urllib.request.build_opener(*handlers)


def destination(asset):
    original = Path(asset['path'])
    assert '..' not in original.parts
    if asset.get('destination_policy'):
        assert asset['destination_policy'] == 'EXISTING_SYMLINK_TARGET_MUST_BE_UNDER_DEV_SHM'
        assert original.is_symlink(), str(original)
    target = original.resolve(strict=False)
    assert target.is_relative_to('/dev/shm') and not target.is_symlink(), str(target)
    return target


def signed_cdn(asset):
    # Actual two-range probe established reuse; the final native SHA remains mandatory.
    with CDN_LOCK:
        cached = CDN_URLS.get(asset['path'])
        if cached and time.monotonic() - cached[0] < 1800:
            return cached[1]
        request = urllib.request.Request(url(asset, int(time.time())),
            headers={'Range': 'bytes=0-31', 'Accept-Encoding': 'identity'})
        try:
            with proxy_opener(True).open(request, timeout=30):
                raise RuntimeError('Pinned large payload requires public CDN redirect')
        except urllib.error.HTTPError as response:
            assert response.code in (302, 307)
            cdn = response.headers['Location']
            parsed = urllib.parse.urlsplit(cdn)
            assert parsed.scheme == 'https' and parsed.hostname.endswith('.hf.co')
        CDN_URLS[asset['path']] = (time.monotonic(), cdn)
        return cdn


def fetch_range(asset, lo):
    hi = min(lo + CHUNK, asset['bytes']) - 1
    for attempt in range(3):
        assert time.monotonic() - START < 7200
        try:
            headers = {'Range': f'bytes={lo}-{hi}', 'Accept-Encoding': 'identity'}
            cdn = signed_cdn(asset)
            request = urllib.request.Request(cdn, headers=headers)
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=45) as response:
                assert response.status == 206
                assert response.headers['Content-Range'] == f"bytes {lo}-{hi}/{asset['bytes']}"
                data = response.read(CHUNK + 1)
            assert len(data) == hi - lo + 1
            return lo, data
        except Exception:
            if attempt == 2:
                raise


def restore(asset):
    target = destination(asset)
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if target.exists():
        assert target.is_file() and sha(target) == asset['sha256'], str(target)
        return dict(path=asset['path'], resolved_path=str(target), bytes=target.stat().st_size,
                    sha256=asset['sha256'], reused=True)
    partial = target.with_name(target.name + '.restore.partial')
    assert not partial.is_symlink()
    if 'bytes' not in asset:
        # Small pinned config/tokenizer files: retain a bad partial for diagnosis.
        if not partial.exists():
            for attempt in range(3):
                try:
                    with proxy_opener().open(url(asset), timeout=30) as response:
                        data = response.read(asset['maximum_bytes'] + 1)
                    assert len(data) <= asset['maximum_bytes']
                    assert hashlib.sha256(data).hexdigest() == asset['sha256']
                    break
                except Exception:
                    if attempt == 2:
                        raise
            assert len(data) <= asset['maximum_bytes']
            with partial.open('xb') as stream:
                stream.write(data); stream.flush(); os.fsync(stream.fileno())
    else:
        state_path = target.with_name(target.name + '.restore.ranges.json')
        if not partial.exists():
            assert not state_path.exists()
            with partial.open('xb') as stream:
                stream.truncate(asset['bytes'])
            write(state_path, dict(expected_sha256=asset['sha256'], completed_ranges={}))
        state = json.loads(state_path.read_text())
        assert state['expected_sha256'] == asset['sha256']
        assert partial.stat().st_size == asset['bytes']
        with partial.open('rb') as stream:
            for lo, record in state['completed_ranges'].items():
                lo = int(lo)
                assert 0 <= lo < asset['bytes'] and lo % CHUNK == 0
                assert record['bytes'] == min(CHUNK, asset['bytes'] - lo)
                stream.seek(lo)
                assert hashlib.sha256(stream.read(record['bytes'])).hexdigest() == record['sha256']
        remaining = iter(lo for lo in range(0, asset['bytes'], CHUNK)
                         if str(lo) not in state['completed_ranges'])
        fd = os.open(partial, os.O_RDWR)
        try:
            with ThreadPoolExecutor(WORKERS) as pool:
                pending = {}
                while True:
                    while len(pending) < WORKERS:
                        lo = next(remaining, None)
                        if lo is None:
                            break
                        pending[pool.submit(fetch_range, asset, lo)] = lo
                    if not pending:
                        break
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        pending.pop(future)
                        lo, data = future.result()
                        assert os.pwrite(fd, data, lo) == len(data)
                        os.fsync(fd)
                        state['completed_ranges'][str(lo)] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                        write(state_path, state)
                        progress = dict(path=asset['path'], complete_bytes=sum(
                            x['bytes'] for x in state['completed_ranges'].values()),
                            total_bytes=asset['bytes'], seconds=time.monotonic() - START)
                        write(PROGRESS, progress)
        finally:
            os.close(fd)
    if 'bytes' in asset:
        assert partial.stat().st_size == asset['bytes']
    assert sha(partial) == asset['sha256'], 'Preserve incorrect partial; never adopt it'
    assert not target.exists()
    partial.rename(target)
    assert sha(asset['path']) == asset['sha256']
    return dict(path=asset['path'], resolved_path=str(target), bytes=target.stat().st_size,
                sha256=asset['sha256'], reused=False)


def main():
    global WORKERS
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=16)
    arguments = parser.parse_args()
    assert 1 <= arguments.workers <= 64
    WORKERS = arguments.workers
    assert os.name == 'posix' and ROOT == Path('/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1')
    manifest = json.loads(MANIFEST.read_text())
    recipe_path = ROOT / 'configs/functional_profiles_20261022_v1/O.json'
    assert sha(recipe_path) == manifest['input_recipe_sha256']
    recipe = json.loads(recipe_path.read_text())
    assets = manifest['assets']
    assert len(assets) == len({a['path'] for a in assets}) == 19
    for asset in assets:
        assert recipe['input_sha256'][asset['path']] == asset['sha256']
        assert asset['repo'] in ('stabilityai/sdxl-turbo', 'fakekungfu/Fusion-Prior-H14_B32_VAE')
        assert len(asset['revision']) == 40 and '..' not in Path(asset['source_path']).parts
    required = sum(a.get('bytes', a.get('maximum_bytes', 0)) for a in assets
                   if not destination(a).exists())
    # At most two generations of completed/in-flight chunk buffers plus 2GiB margin.
    assert shutil.disk_usage('/dev/shm').free >= required + (2 * WORKERS * CHUNK) + (2 << 30)
    files = []
    # Configs and small priors first; largest public payload last.
    for asset in sorted(assets, key=lambda a: a.get('bytes', a.get('maximum_bytes', 0))):
        assert time.monotonic() - START < 7200
        record = restore(asset); files.append(record)
        print(json.dumps(record), flush=True)
    assert all(sha(a['path']) == a['sha256'] for a in assets)
    write(OUT, dict(status='ALL19_PINNED_PUBLIC_RUNTIME_ASSETS_NATIVE_SHA_VERIFIED',
        UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), source_sha256=sha(__file__),
        manifest_sha256=sha(MANIFEST), files=files, seconds=time.monotonic() - START,
        download_workers=WORKERS, maximum_two_generation_chunk_buffer_bytes=2 * WORKERS * CHUNK,
        GPU_calls=0, private_EEG_downloaded=False, historical_receipts_changed=False,
        private_T_latent_restored=False, formal_training_admitted=False,
        RAM_is_volatile=True, no_local_large_payloads=True))


if __name__ == '__main__':
    main()
