"""Finite parallel HTTP ranges, preserving original partials and exact SHA pins."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,concurrent.futures,datetime,hashlib,json,shutil,time,urllib.error,urllib.request,traceback
from pathlib import Path

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main(manifest_path):
    spec=json.loads(manifest_path.read_bytes());root=Path(spec['output_root']);work=root.parent/'public_ranges_v4';assert not work.exists();work.mkdir()
    start=time.monotonic();deadline=start+7200;receipts=[]
    def write(path,value):
        with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    try:
        assert shutil.disk_usage(root).free>8<<30
        clinical=root/'labram-base.pth';assert sha(clinical)=='7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c'
        for entry in spec['weights']:
            if entry['name']=='labram-base.pth':continue
            destination=root/entry['name']
            if destination.exists():assert destination.stat().st_size==entry['bytes'] and sha(destination)==entry['sha256'];receipts.append(dict(name=entry['name'],sha256=entry['sha256'],reused_complete=True));continue
            folder=work/entry['name'];folder.mkdir();original=root/(entry['name']+'.partial');prefix=folder/'preserved_prefix.bin'
            prefix_bytes=0
            if original.exists():
                prefix_bytes=original.stat().st_size;assert 0<prefix_bytes<entry['bytes']
                with original.open('rb') as source,prefix.open('xb') as dest:shutil.copyfileobj(source,dest,1<<20)
                assert prefix.stat().st_size==prefix_bytes and sha(prefix)==sha(original)
                # Reuse only a consecutive prefix of the previous verified-size
                # range files. The final whole-file pinned SHA remains mandatory.
                previous=root.parent/'public_ranges_v3'/entry['name'];reused=[]
                index=0
                with prefix.open('ab') as dest:
                    while True:
                        prior=previous/f'part_{index:04d}.bin'
                        expected=min(8<<20,entry['bytes']-prefix_bytes)
                        if not prior.exists():break
                        assert expected>0 and prior.stat().st_size==expected
                        with prior.open('rb') as source:shutil.copyfileobj(source,dest,1<<20)
                        reused.append(dict(path=str(prior),bytes=expected,sha256=sha(prior)));prefix_bytes+=expected;index+=1
                write(folder/'prefix_receipt.json',dict(original=str(original),bytes=prefix_bytes,sha256=sha(prefix),original_unchanged=True,reused_previous_chunks=reused))
            ranges=[(i,lo,min(lo+(64<<20),entry['bytes'])) for i,lo in enumerate(range(prefix_bytes,entry['bytes'],64<<20))]
            def download(item):
                index,lo,hi=item;assert time.monotonic()<deadline
                for attempt in (1,2,3):
                    # Try the official endpoint first, then the same immutable
                    # revision through the existing mirror after a recorded wait.
                    url=entry['url'] if attempt==1 else entry['url'].replace('https://huggingface.co/','https://hf-mirror.com/')
                    partial=folder/f'part_{index:04d}_attempt{attempt}.partial'
                    try:
                        request=urllib.request.Request(url,headers={'Range':f'bytes={lo}-{hi-1}','Accept-Encoding':'identity','User-Agent':'EEG-author-checkpoint-verified-ranges-v3'})
                        with urllib.request.urlopen(request,timeout=90) as response,partial.open('xb') as out:
                            assert response.status==206 and response.headers.get('Content-Range')==f'bytes {lo}-{hi-1}/{entry["bytes"]}',dict(status=response.status,range=response.headers.get('Content-Range'))
                            received=0
                            while received<hi-lo:
                                assert time.monotonic()<deadline
                                block=response.read(min(1<<18,hi-lo-received));assert block
                                out.write(block);received+=len(block)
                            assert not response.read(1)
                        final=folder/f'part_{index:04d}.bin';partial.rename(final)
                        return dict(index=index,start=lo,end=hi,path=str(final),bytes=final.stat().st_size,sha256=sha(final))
                    except BaseException as error:
                        write(folder/f'part_{index:04d}_attempt{attempt}_failure.json',dict(error=repr(error),traceback=traceback.format_exc(),partial_preserved=True))
                        if attempt==3:raise
                        wait=60
                        if isinstance(error,urllib.error.HTTPError):
                            retry=error.headers.get('Retry-After','60')
                            if retry.isdigit():wait=max(60,min(300,int(retry)))
                        assert time.monotonic()+wait<deadline;time.sleep(wait)
            completed=[]
            for item in ranges:
                if completed:time.sleep(20)
                completed.append(download(item));progress=dict(status='EXACT_REVISION_RATE_LIMITED_LARGE_RANGES_IN_PROGRESS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),name=entry['name'],preserved_prefix_bytes=prefix_bytes,completed_new_bytes=sum(x['bytes'] for x in completed),expected_bytes=entry['bytes'],completed_chunks=len(completed),total_chunks=len(ranges),seconds=time.monotonic()-start,GPU_jobs=0)
                pending=work/'progress.pending.json';pending.write_text(json.dumps(progress,indent=2)+'\n',encoding='utf8');os.replace(pending,work/'progress.json')
            assembled=folder/'assembled_verified.partial'
            with assembled.open('xb') as out:
                if prefix.exists():
                    with prefix.open('rb') as f:shutil.copyfileobj(f,out,1<<20)
                for item in sorted(completed,key=lambda x:x['index']):
                    p=Path(item['path']);assert sha(p)==item['sha256']
                    with p.open('rb') as f:shutil.copyfileobj(f,out,1<<20)
            assert assembled.stat().st_size==entry['bytes'] and sha(assembled)==entry['sha256']
            os.link(assembled,destination)
            receipt=dict(name=entry['name'],path=str(destination),bytes=entry['bytes'],sha256=entry['sha256'],revision=entry['revision'],prefix_bytes=prefix_bytes,chunks=completed)
            write(folder/'completion.json',receipt);receipts.append(receipt)
        write(work/'completion.json',dict(status='ALL_PINNED_PUBLIC_WEIGHT_BYTES_VERIFIED_NOT_MODEL_LOAD_PROOF',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),receipts=receipts,seconds=time.monotonic()-start,GPU_jobs=0,model_load_not_yet_performed=True))
    except BaseException as error:
        write(work/'failure.json',dict(status='PUBLIC_RANGE_TRANSPORT_FAILURE_PRESERVED',error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-start,GPU_jobs=0));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);main(p.parse_args().manifest)
