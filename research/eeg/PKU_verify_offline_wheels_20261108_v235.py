"""Bulk package hashing belongs to the allocated CPU node, not login."""
import hashlib,json,os,socket
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
root=Path('/dfs/share/maleiLab/wwt/eeg')
manifest=json.loads((root/'manifests/offline_wheels_20261108_v235.json').read_text())
assert manifest['status']=='COMPLETE_ALL24_OFFLINE_WHEELS_STREAMED_SHA256_PASS'
for item in manifest['files']:
    p=Path(item['remote_path']);assert p.parent==root/'package_cache/wheels_v235'
    assert p.stat().st_size==item['actual_bytes']
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(4<<20),b''):h.update(chunk)
    assert h.hexdigest()==item['sha256'],p.name
print(json.dumps(dict(status='PASS_ALL24_WHEELS_COMPUTE_NODE_SHA256',job_id=os.environ['SLURM_JOB_ID'],total_bytes=manifest['total_bytes'])),flush=True)
