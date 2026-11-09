"""Acquire the exact official SDXL base snapshot on server, without HF cache duplication."""
from pathlib import Path
import concurrent.futures as futures
import fcntl
import hashlib
import json
import os
import shutil
import time
import traceback
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parent
OUT = Path('/dev/shm/eeg_sota_autoresearch_20261018_v1/sdxl_base_46216598')
START = time.monotonic()
CHUNK = 4 * 1024**2
MANIFEST = json.loads((ROOT / 'audit/sdxl_base_selected_assets.json').read_text())
assert MANIFEST['revision'] == '462165984030d82259a11f4367a4eed129e94a7b'


def write(path, data):
    temp = path.with_name(path.name + '.pending')
    temp.write_text(json.dumps(data, indent=2) + '\n'); os.replace(temp, path)


def digest(path, algorithm='sha256', git_blob=False):
    h = hashlib.new(algorithm)
    if git_blob:
        h.update(('blob ' + str(path.stat().st_size) + '\0').encode())
    with path.open('rb') as file:
        for b in iter(lambda:file.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def fetch(asset, lo):
    hi = min(lo+CHUNK, asset['size'])-1
    url = ('https://hf-mirror.com/' + MANIFEST['repository'] + '/resolve/' +
           MANIFEST['revision'] + '/' + asset['path'])
    for attempt in range(4):
        assert time.monotonic()-START < 7200
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            req = urllib.request.Request(url, headers={'Range':f'bytes={lo}-{hi}',
                  'Accept-Encoding':'identity', 'User-Agent':'EEG-SDXL-source/1.0'})
            with opener.open(req, timeout=12) as response:
                if 'lfs' in asset:
                    assert response.status == 206
                    assert response.headers.get('Content-Range') == f'bytes {lo}-{hi}/{asset["size"]}'
                b = response.read(hi-lo+2)
            assert len(b) == hi-lo+1
            return lo,b
        except Exception as exc:
            if attempt == 3:
                raise
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 429:
                retry = exc.headers.get('Retry-After', '')
                time.sleep(min(60, max(10, int(retry) if retry.isdigit() else 10*(attempt+1))))


def verify(path, asset):
    assert path.stat().st_size == asset['size']
    if 'lfs' in asset:
        assert digest(path) == asset['lfs']['oid'], path
    else:
        assert digest(path, 'sha1', git_blob=True) == asset['oid'], path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT/'acquisition.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert shutil.disk_usage(OUT).free > sum(a['size'] for a in MANIFEST['assets']) + 4*1024**3
    acquired=[]
    for asset in MANIFEST['assets']:
        target = OUT/asset['path']; target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            verify(target,asset)
        else:
            part=target.with_name(target.name+'.download')
            journal=target.with_name(target.name+'.chunks.jsonl')
            if not part.exists():
                with part.open('xb') as f:f.truncate(asset['size'])
            assert part.stat().st_size == asset['size'] and not part.is_symlink()
            complete=set()
            if journal.exists():
                with part.open('rb') as f:
                    for line in journal.read_text().splitlines():
                        row=json.loads(line);f.seek(row['start']);b=f.read(row['bytes'])
                        assert hashlib.sha256(b).hexdigest()==row['sha256'];complete.add(row['start'])
            fd=os.open(part,os.O_RDWR)
            try:
                with futures.ThreadPoolExecutor(max_workers=4) as pool,journal.open('a') as log:
                    source=iter(lo for lo in range(0,asset['size'],CHUNK) if lo not in complete)
                    pending={}
                    def add():
                        lo=next(source,None)
                        if lo is not None:pending[pool.submit(fetch,asset,lo)]=lo
                    for _ in range(8):add()
                    while pending:
                        ready,_=futures.wait(pending,return_when=futures.FIRST_COMPLETED)
                        for job in ready:
                            pending.pop(job);lo,b=job.result();assert os.pwrite(fd,b,lo)==len(b)
                            log.write(json.dumps(dict(start=lo,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))+'\n');log.flush()
                            complete.add(lo);add()
                        write(OUT/'progress.json',dict(stage='DOWNLOADING',asset=asset['path'],pid=os.getpid(),
                              complete_chunks=len(complete),total_chunks=(asset['size']+CHUNK-1)//CHUNK,
                              seconds=time.monotonic()-START))
                os.fsync(fd)
            finally:os.close(fd)
            verify(part,asset);os.replace(part,target)
        acquired.append(dict(path=str(target),bytes=target.stat().st_size,sha256=digest(target),verified=True))
        print(json.dumps(dict(stage='ASSET_VERIFIED',path=str(target),bytes=target.stat().st_size)),flush=True)
    write(OUT/'completion.json',dict(status='COMPLETE_OFFICIAL_SNAPSHOT',repository=MANIFEST['repository'],
          revision=MANIFEST['revision'],files=acquired,seconds=time.monotonic()-START,
          local_large_files_created=0,generation_calls=0,training_updates=0,
          storage='volatile server RAM mount; public weights can be re-acquired; all learned checkpoints and images remain on persistent server data disk'))


def acquire_snapshot(manifest, output):
    """Reuse the verified transport for another pinned public runtime snapshot."""
    global MANIFEST, OUT, START
    assert manifest['repository'] and len(manifest['revision'])==40 and manifest['assets']
    MANIFEST=manifest;OUT=Path(output);START=time.monotonic()
    assert OUT.is_relative_to('/dev/shm/eeg_sota_autoresearch_20261018_v1')
    main()


if __name__=='__main__':
    try:main()
    except Exception as exc:
        OUT.mkdir(parents=True,exist_ok=True)
        write(OUT/'failure.json',dict(status='FAILED_RESUMABLE',type=type(exc).__name__,
              message=str(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-START))
        raise
