"""One ephemeral public signed-CDN stream to a private exact-SHA checkpoint."""
import datetime,hashlib,json,os,sys,time,traceback,urllib.parse,urllib.request
from pathlib import Path
q=json.loads(sys.stdin.buffer.readline());url=q.pop('signed_public_url');host=urllib.parse.urlparse(url).hostname;assert host.endswith('.hf.co')
M=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/multiscale_EEG_20261028_v1');B=M/'public_CDN_direct_v5';B.mkdir();start=time.monotonic();expected=1243116346;digest='4f6151aad91c94e03e224dd7358d8389fa05301c0c291279566998efb54b0ecb'
def write(name,value):
    p=B/name
    with p.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
try:
    p=B/'var_d16.received.partial';total=0;h=hashlib.sha256();threshold=32<<20
    request=urllib.request.Request(url,headers={'Range':f'bytes=0-{expected-1}','Accept-Encoding':'identity','User-Agent':'EEG-exact-public-CDN-one-stream'})
    with urllib.request.urlopen(request,timeout=90) as response,p.open('xb') as out:
        assert response.status in (200,206)
        if response.status==206:assert response.headers.get('Content-Range')==f'bytes 0-{expected-1}/{expected}'
        while total<expected:
            assert time.monotonic()-start<1800
            block=response.read(min(1<<20,expected-total));assert block;out.write(block);h.update(block);total+=len(block)
            if total>=threshold:
                progress=dict(status='EXACT_PUBLIC_CDN_ONE_STREAM_PROGRESS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),bytes=total,expected_bytes=expected,seconds=time.monotonic()-start,final_host=host,signed_URL_query_persisted=False)
                pending=B/'progress.pending.json';pending.write_text(json.dumps(progress,indent=2)+'\n',encoding='utf8');os.replace(pending,B/'progress.json');threshold+=32<<20
        assert not response.read(1)
    assert total==expected and h.hexdigest()==digest
    target=B/'var_d16.verified.pth';os.link(p,target)
    write('completion.json',dict(status='OFFICIAL_IMMUTABLE_VAR_PUBLIC_CDN_BYTES_FULL_SHA_VERIFIED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),file=dict(path=str(target),bytes=total,sha256=digest),source_revision='6d0ee6598a42f75079aa79e58b012b57b284c91b',source_repository='FoundationVision/var',final_host=host,seconds=time.monotonic()-start,signed_public_URL_query_persisted=False,public_v4_worker_untouched=True,GPU_jobs=0))
except BaseException as error:
    # Do not serialize URL-bearing exception representations or request data.
    write('failure.json',dict(status='PUBLIC_SIGNED_CDN_STREAM_FAILURE_PRESERVED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),error_type=type(error).__name__,HTTP_code=getattr(error,'code',None),received_bytes=locals().get('total'),seconds=time.monotonic()-start,signed_public_URL_query_persisted=False));raise RuntimeError('Public signed CDN stream failed; private prefix preserved') from None
