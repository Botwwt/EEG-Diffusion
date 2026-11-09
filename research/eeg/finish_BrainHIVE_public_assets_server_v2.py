"""Resume verified public asset transfer through the temporary private relay."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

ASSETS = Path('/root/brainhive_reproduction_20261020_v1/assets')
CHUNK = 8 << 20


def sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def write(path, value):
    tmp = path.with_name(path.name + '.pending')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def main():
    started = time.monotonic()
    plan = json.loads((ASSETS / 'transfer_plan.json').read_text())
    for asset in plan['assets']:
        path = ASSETS / asset['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        ready = Path(str(path) + '.ready.json')
        if path.exists():
            assert path.stat().st_size == asset['bytes'] and sha(path) == asset['sha256']
            if not ready.exists():
                write(ready, asset | dict(full_native_SHA_verified=True))
            continue
        pending = Path(str(path) + '.partial')
        state_path = Path(str(path) + '.range_state.json')
        if state_path.exists():
            state = json.loads(state_path.read_text())
            assert state['expected_sha256'] == asset['sha256']
        else:
            prefix = pending.stat().st_size if pending.exists() else 0
            assert 0 <= prefix <= asset['bytes']
            state = dict(expected_sha256=asset['sha256'], initial_sequential_prefix_bytes=prefix,
                         initial_prefix_source='interrupted sequential RAM-to-SFTP transfer; full native SHA required',
                         completed_ranges={})
            write(state_path, state)
        fd = os.open(pending, os.O_CREAT | os.O_RDWR, 0o600)
        os.ftruncate(fd, asset['bytes'])
        starts = [lo for lo in range(state['initial_sequential_prefix_bytes'], asset['bytes'], CHUNK)
                  if str(lo) not in state['completed_ranges']]

        def fetch(lo):
            hi = min(lo + CHUNK, asset['bytes']) - 1
            url = ('https://huggingface.co/' + asset['repo'] + '/resolve/' + asset['revision']
                   + '/' + asset['source_path'] + '?download=true&server_resume_range=' + str(lo))
            errors = []
            for attempt in range(3):
                try:
                    assert time.monotonic() - started < 7200
                    opener = urllib.request.build_opener(urllib.request.ProxyHandler({'https': 'http://127.0.0.1:33343'}))
                    request = urllib.request.Request(url, headers={'Range': f'bytes={lo}-{hi}', 'Accept-Encoding': 'identity'})
                    with opener.open(request, timeout=60) as response:
                        assert response.status == 206
                        assert response.headers['Content-Range'] == f"bytes {lo}-{hi}/{asset['bytes']}"
                        data = response.read(CHUNK + 1)
                    assert len(data) == hi - lo + 1
                    return lo, data
                except Exception as exc:
                    errors.append(type(exc).__name__ + ':' + str(exc)[:120])
            raise RuntimeError('Public range failed ' + str(lo) + ' ' + repr(errors))

        try:
            todo = iter(starts)
            futures = {}
            with ThreadPoolExecutor(4) as pool:
                while True:
                    while len(futures) < 4:
                        lo = next(todo, None)
                        if lo is None:
                            break
                        futures[pool.submit(fetch, lo)] = lo
                    if not futures:
                        break
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        futures.pop(future)
                        lo, data = future.result()
                        assert os.pwrite(fd, data, lo) == len(data)
                        state['completed_ranges'][str(lo)] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                        write(state_path, state)
                        complete = state['initial_sequential_prefix_bytes'] + sum(x['bytes'] for x in state['completed_ranges'].values())
                        progress = dict(stage='SERVER_PUBLIC_ASSET_TRANSFER', path=asset['path'], completed_bytes=complete,
                                        total_bytes=asset['bytes'], seconds=time.monotonic() - started, workers=4)
                        write(ASSETS / 'server_transfer_progress.json', progress)
                        print(json.dumps(progress), flush=True)
            os.fsync(fd)
        finally:
            os.close(fd)
        assert sha(pending) == asset['sha256'], 'Do not adopt an incorrect partial payload'
        pending.rename(path)
        write(ready, asset | dict(full_native_SHA_verified=True))
    for name in ('fusion_encoder/config.json', 'proj/config.json', 'attn_adapter/config.json'):
        assert (ASSETS / 'prior_H14' / name).exists()
    write(ASSETS / 'transfer_complete.json', dict(status='ALL15_PUBLIC_PAYLOADS_FULL_NATIVE_SHA_VERIFIED',
        assets=15, bytes=sum(a['bytes'] for a in plan['assets']), seconds=time.monotonic()-started,
        local_large_files_written=False, temporary_private_public_only_relay=True))
    print('PUBLIC_ASSETS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
