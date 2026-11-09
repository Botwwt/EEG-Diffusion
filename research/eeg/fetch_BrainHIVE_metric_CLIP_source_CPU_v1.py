"""Fetch a small, Git-blob-verified official CLIP source pin, never model data."""
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parent


def api(path):
    request=urllib.request.Request('https://api.github.com/repos/openai/CLIP/'+path,
        headers={'User-Agent':'EEG-reproduction-source-audit','Accept':'application/vnd.github+json'})
    with urllib.request.urlopen(request,timeout=35) as response:
        assert response.status==200 and 'json' in response.headers['Content-Type']
        data=response.read(1<<20)
    assert len(data)<1<<20
    return json.loads(data)


def main():
    commit=api('commits/main')['sha']
    destination=ROOT/'reproduction_sources'/('OpenAI_CLIP_'+commit[:12])
    assert not destination.exists(),'Preserve the existing source fetch'
    destination.mkdir(parents=True)
    def fetch(name):
        item=api('contents/'+name+'?ref='+commit)
        data=base64.b64decode(item['content']);assert len(data)==item['size']
        blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        assert blob==item['sha']
        path=destination/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        return dict(path=name,bytes=len(data),git_blob=blob,sha256=hashlib.sha256(data).hexdigest())
    with ThreadPoolExecutor(3) as pool:files=list(pool.map(fetch,['clip/model.py','clip/clip.py','LICENSE']))
    receipt=dict(repository='https://github.com/openai/CLIP',commit=commit,files=files,
        use='Original BrainHIVE CLIP-L14 image evaluation only',model_weights_acquired=False)
    (destination/'_fetch_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    path=ROOT/'audit/BrainHIVE_CLIP_metric_source_pin_20261020_v1.json'
    assert not path.exists();path.write_text(json.dumps(receipt|dict(local_source_root=str(destination)),indent=2)+'\n')
    print(json.dumps(dict(commit=commit,files=len(files),source_bytes=sum(f['bytes'] for f in files),path=str(destination))))


if __name__=='__main__':main()
