"""Transport-only retry of v1; immutable author digests and bytes are unchanged."""
import hashlib,json,runpy,sys,urllib.request
from pathlib import Path
root=Path(__file__).resolve().parent
worker=root/'prepare_multiscale_public_assets_CPU_20261028_v1.py'
assert hashlib.sha256(worker.read_bytes()).hexdigest()=='0caa824e08aedfa88f6b7dbbe970741a47be5b389d09c3969231ed50633a4d0c'
original=urllib.request.urlopen
def public_transport(request,*args,**kwargs):
    url=request.full_url if isinstance(request,urllib.request.Request) else request
    if url.startswith('https://huggingface.co/FoundationVision/var/resolve/6d0ee6598a42f75079aa79e58b012b57b284c91b/'):
        request=urllib.request.Request(url.replace('https://huggingface.co/','https://hf-mirror.com/'),headers=dict(request.header_items()))
    kwargs['timeout']=120
    return original(request,*args,**kwargs)
urllib.request.urlopen=public_transport
runpy.run_path(str(worker),run_name='__main__')
