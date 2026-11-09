"""Read-only public source provenance. Never execute the author's code."""
import concurrent.futures,hashlib,json,time,urllib.request
from pathlib import Path,PurePosixPath
L=Path(__file__).resolve().parent;O=L/'audit/EEGRFusion_public_source_review_20261101_v1';inventory=json.loads((O/'inventory.json').read_bytes());code=O/'source';code.mkdir(exist_ok=False);start=time.monotonic()
assert sum(v['size'] for v in inventory['text_candidates'])<8<<20
def fetch(row):
    name=row['path'];parts=PurePosixPath(name);assert not parts.is_absolute() and '..' not in parts.parts
    assert row['size']<3<<20
    url='https://raw.githubusercontent.com/hcy1026/EEGRFusion/'+inventory['commit']+'/'+name
    request=urllib.request.Request(url,headers={'User-Agent':'EEG-research-static-source-audit'})
    with urllib.request.urlopen(request,timeout=25) as response:b=response.read((3<<20)+1)
    assert len(b)==row['size'] and hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==row['sha']
    path=code/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b)
    if name.endswith('.ipynb'):
        notebook=json.loads(b);cells=[]
        for i,c in enumerate(notebook['cells']):
            if c['cell_type']=='code':cells.append('# ORIGINAL_NOTEBOOK_CELL '+str(i)+'\n'+''.join(c['source'])+'\n')
        target=O/'notebook_code'/parts.with_suffix('.code.py');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(('\n'.join(cells)).encode())
    return dict(path=name,bytes=len(b),git_blob_SHA1=row['sha'],sha256=hashlib.sha256(b).hexdigest(),URL=url)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(fetch,inventory['text_candidates']))
result=dict(status='COMPLETE_PINNED_EEGRFUSION_ALL_TEXT_SOURCE_RETRIEVAL_NOT_EXECUTION',commit=inventory['commit'],tree=inventory['tree'],text_files=len(records),total_bytes=sum(v['bytes'] for v in records),files=records,seconds=time.monotonic()-start,author_code_imported=False,author_code_executed=False,weights_or_data_downloaded=False,models_or_metrics_reproduced=False,new_GPU_jobs=0)
(O/'source_provenance.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in result.items() if k!='files'}))
