"""Slurm CPU: offline two-dependency install, public SHA, original teacher input join."""
import csv,json,os,socket,subprocess,sys,time,traceback
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
ROOT=Path('/dfs/share/maleiLab/wwt/eeg');out=ROOT/'data/native_training_v239';out.mkdir(mode=0o700);start=time.monotonic()
def write(name,v):
    p=out/name;t=p.with_suffix('.pending');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
try:
    from PKU_HIVE_native_classes_20261109_v239 import sha
    m=json.loads((ROOT/'manifests/native_public_assets_20261109_v239.json').read_text());assert m['status']=='COMPLETE_NATIVE_TEACHER_PUBLIC_ASSETS_STREAMED'
    for r in m['files']:
        p=ROOT/r['relative_path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
    wheels=[str(ROOT/r['relative_path']) for r in m['files'] if r['relative_path'].endswith('.whl')];assert len(wheels)==2
    subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','--no-index','--no-deps']+wheels,check=True)
    import numpy as np
    import torch
    import pyarrow.parquet as pq
    from PKU_HIVE_native_classes_20261109_v239 import classes,teacher
    torch.set_num_threads(4)
    with (ROOT/'data/full_T_recovery_20261108_v238/image_manifest.csv').open() as f:
        rows=sorted([r for r in csv.DictReader(f) if r['dataset_split']=='train'],key=lambda r:int(r['array_row']))
    assert [int(r['array_row']) for r in rows]==list(range(16540))
    ids=[Path(r['stimulus_id']).stem for r in rows];positions={n:i for i,n in enumerate(ids)};assert len(positions)==16540
    cfg=json.loads((ROOT/'code/HIVE_fusion_config_v239.json').read_text());embeddings={};schemas=[]
    for key,dim in cfg['proj_meta'].items():
        values=torch.empty(16540,dim,dtype=torch.float32);seen=set()
        paths=sorted((ROOT/'assets/native_teacher_v239/features').glob('things_train_'+key+'-part-*.parquet'));assert len(paths)==2
        for p in paths:
            count=0
            for batch in pq.ParquetFile(p).iter_batches(batch_size=256,columns=['image_id','emb']):
                d=batch.to_pydict();names=[str(x) for x in d['image_id']];assert len(set(names))==len(names) and not set(names)&seen
                assert set(names)<=positions.keys()
                a=np.asarray(d['emb'],dtype=np.float32).reshape(len(names),-1);assert a.shape==(len(names),dim) and np.isfinite(a).all()
                values[[positions[n] for n in names]]=torch.from_numpy(a);seen.update(names);count+=len(names)
            schemas.append(dict(file=p.name,key=key,rows=count,dimension=dim))
        assert seen==set(ids);embeddings[key]=values
    torch.save(embeddings,out/'train_visual_embeddings.pt')
    write('train_image_ids.json',ids)
    split=json.loads((ROOT/'data/recovery_20261108_v236/current_split_manifest.json').read_text());T=split['image_rows']['T']
    target=np.load(ROOT/'data/full_T_recovery_20261108_v238/T_gallery_F.npy',mmap_mode='r',allow_pickle=False)
    probes=np.linspace(0,13229,8,dtype=np.int64);chosen=[T[int(i)] for i in probes]
    model=teacher(classes())
    with torch.inference_mode():v=model(**{k:x[chosen] for k,x in embeddings.items()})
    reference=torch.from_numpy(np.array(target[probes]));cos=torch.nn.functional.cosine_similarity(v,reference)
    relative=(v-reference).norm(dim=-1)/reference.norm(dim=-1)
    assert float(cos.min())>.99998 and float(relative.max())<.005,('native teacher mismatch',cos,relative)
    result=dict(status='COMPLETE_NATIVE_HIVE_TEACHER_TRAIN_INPUTS_RESTORED',job_id=os.environ['SLURM_JOB_ID'],hostname=socket.gethostname(),
        seconds=time.monotonic()-start,public_assets_SHA_verified=9,training_images=16540,joined_images_per_modality=16540,
        source_F_original_submodule=True,CPU_F_minimum_cosine=float(cos.min()),CPU_F_maximum_relative_L2=float(relative.max()),
        deterministic_cache_not_used_as_training_teacher=True,stochastic_teacher_input_available=True,
        visual_embeddings_sha256=sha(out/'train_visual_embeddings.pt'),schemas=schemas,official_test_read=False,new_updates=0)
    write('completion.json',result);print(json.dumps(result),flush=True)
except BaseException as e:
    write('failure.json',dict(status='FAILED_PRESERVED',error=repr(e),traceback=traceback.format_exc()));raise
