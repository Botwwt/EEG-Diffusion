"""One rate-respecting mirror-tail stream; immutable v4 prefixes are reused."""
import datetime,hashlib,json,os,time,urllib.error,urllib.request
from pathlib import Path
M=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/multiscale_EEG_20261028_v1');B=M/'public_mirror_tail_v6';assert not B.exists();B.mkdir();old=M/'public_ranges_v4/var_d16.pth';public=M/'public_assets/var_d16.pth';start=time.monotonic();size=1243116346;expected='4f6151aad91c94e03e224dd7358d8389fa05301c0c291279566998efb54b0ecb';width=64<<20;count=(size+width-1)//width
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def progress(**kwargs):
    value=dict(status='ONE_MIRROR_TAIL_WITH_ORIGINAL_V4_PREFIX_REUSE',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.monotonic()-start,**kwargs);pending=B/'progress.pending.json';pending.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8');os.replace(pending,B/'progress.json')
def existing(index):
    length=min(width,size-index*width)
    for path in [old/f'part_{index:04d}.bin',B/f'part_{index:04d}.bin']:
        if path.is_file() and path.stat().st_size==length:return path
    return None
try:
    for index in range(count-1,-1,-1):
        assert time.monotonic()-start<3600
        if public.exists():assert public.stat().st_size==size and sha(public)==expected;break
        if existing(index) is not None:continue
        lo=index*width;hi=min(size,lo+width)
        for attempt in (1,2,3):
            path=B/f'part_{index:04d}_attempt{attempt}.partial';received=0
            try:
                request=urllib.request.Request('https://hf-mirror.com/FoundationVision/var/resolve/6d0ee6598a42f75079aa79e58b012b57b284c91b/var_d16.pth',headers={'Range':f'bytes={lo}-{hi-1}','Accept-Encoding':'identity','User-Agent':'EEG-public-exact-one-mirror-tail'})
                with urllib.request.urlopen(request,timeout=90) as response,path.open('xb') as out:
                    assert response.status==206 and response.headers.get('Content-Range')==f'bytes {lo}-{hi-1}/{size}'
                    while received<hi-lo:
                        assert time.monotonic()-start<3600
                        data=response.read(min(1<<20,hi-lo-received));assert data;out.write(data);received+=len(data)
                        progress(active_index=index,active_range_bytes=received,expected_range_bytes=hi-lo,complete_indices=[i for i in range(count) if existing(i) is not None])
                    assert not response.read(1)
                path.rename(B/f'part_{index:04d}.bin');break
            except BaseException as error:
                write(B/f'part_{index:04d}_attempt{attempt}_failure.json',dict(error_type=type(error).__name__,HTTP_code=getattr(error,'code',None),partial_bytes=received,preserved=True))
                if attempt==3:raise RuntimeError('Mirror-tail transport failed after bounded rate-respecting attempts') from None
                retry=error.headers.get('Retry-After','60') if isinstance(error,urllib.error.HTTPError) else '60';pause=max(60,min(300,int(retry))) if retry.isdigit() else 60
                progress(active_index=index,backoff_seconds=pause,HTTP_code=getattr(error,'code',None));time.sleep(pause)
        if all(existing(i) is not None for i in range(count)):break
        time.sleep(5)
    if public.exists():
        assert public.stat().st_size==size and sha(public)==expected;target=public;sources=[]
    else:
        assert all(existing(i) is not None for i in range(count));p=B/'assembled.partial';sources=[]
        with p.open('xb') as out:
            for i in range(count):
                source=existing(i);sources.append(dict(index=i,path=str(source),bytes=source.stat().st_size,sha256=sha(source)))
                with source.open('rb') as stream:
                    for data in iter(lambda:stream.read(1<<20),b''):out.write(data)
        assert p.stat().st_size==size and sha(p)==expected;target=B/'var_d16.verified.pth';os.link(p,target)
    write(B/'completion.json',dict(status='OFFICIAL_VAR_FULL_SHA_READY_FROM_MIRROR_TAIL_AND_V4_PREFIX',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),file=dict(path=str(target),bytes=size,sha256=expected),sources=sources,seconds=time.monotonic()-start,original_v4_worker_untouched=True,GPU_jobs=0))
except BaseException as error:
    write(B/'failure.json',dict(status='MIRROR_TAIL_FAILED_PREFIXES_RETAINED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),error_type=type(error).__name__,seconds=time.monotonic()-start));raise RuntimeError('Public mirror-tail failed; all prefixes preserved') from None
