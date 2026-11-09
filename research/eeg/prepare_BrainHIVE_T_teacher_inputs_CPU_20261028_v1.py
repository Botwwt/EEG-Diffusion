"""Export only authorized T image-feature rows for a frozen teacher diagnostic."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def ref(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))

def checked(item):
    p=Path(item['path']);assert p.is_file() and sha(p)==item['sha256']
    if 'bytes' in item:assert p.stat().st_size==item['bytes']
    return p

def run(spec):
    start=time.monotonic()
    import torch
    torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    root=Path('/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1')
    d33=json.loads(checked(spec['D33_spec']).read_bytes())
    index=checked(d33['T_index']);records=[json.loads(x) for x in index.read_text().splitlines()]
    assert len(records)==13230 and all(x['row']==i and x['partition']=='T' for i,x in enumerate(records))
    rows=[int(x['array_row']) for x in records]
    assert all(int(x['source_row'])==r for x,r in zip(records,rows))
    split=json.loads(checked(d33['split']).read_bytes())
    assert len(set(rows))==13230 and set(rows)==set(split['image_rows']['T'])
    assert set(rows).isdisjoint(split['image_rows']['D']+split['image_rows']['V'])
    destination=Path(spec['output']);destination.mkdir(parents=True,exist_ok=True)
    complete=destination/'completion.json'
    if complete.exists():
        done=json.loads(complete.read_bytes());checked(done['T_visual_embeddings']);print(json.dumps(done));return
    source=checked(spec['visual_embeddings'])
    inputs=torch.load(source,map_location='cpu',weights_only=True,mmap=True)
    assert {k:tuple(v.shape) for k,v in inputs.items()}=={
        'CLIP-ViT-B-32-laion2B-s34B-b79K':(16540,512),
        'CLIP-ViT-H-14-laion2B-s32B-b79K':(16540,1024),'vae':(16540,1024)}
    ids=torch.tensor(rows,dtype=torch.long)
    T={k:v.index_select(0,ids).clone().contiguous() for k,v in inputs.items()}
    for k,v in T.items():
        assert torch.isfinite(v).all() and v.untyped_storage().nbytes()==v.numel()*v.element_size()
        assert torch.equal(v,inputs[k].index_select(0,ids))
    output=destination/'all13230_T_visual_embeddings.pt';pending=output.with_name(output.name+'.pending')
    assert not output.exists() and not pending.exists()
    with pending.open('xb') as f:torch.save(T,f);f.flush();os.fsync(f.fileno())
    os.rename(pending,output)
    again=torch.load(output,map_location='cpu',weights_only=True,mmap=True)
    assert all(torch.equal(again[k],v) for k,v in T.items())
    value=dict(status='D36_ALL13230_T_VISUAL_INPUT_ROWS_EXPORTED_CPU_VERIFIED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               source=ref(source),T_index=ref(index),source_spec=spec,T_visual_embeddings=ref(output),rows=13230,concepts=1323,
               no_nonT_feature_values_exported=True,all_tensor_storages_exact_T_only=True,
               T_shapes={k:list(v.shape) for k,v in T.items()},source_order_exactly_T_index=True,
               new_models_fit=0,new_GPU_jobs=0,new_images=0,CUDA_initialized=torch.cuda.is_initialized(),seconds=time.monotonic()-start)
    with complete.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    assert not torch.cuda.is_initialized();print(json.dumps(value))

if __name__=='__main__':run(json.loads(sys.stdin.buffer.readline()))
