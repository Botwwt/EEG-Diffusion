"""Slurm audit: true full T F/CLIP targets and separately identified LaBraM OOF."""
import csv,hashlib,json,os,socket,time,traceback
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
root=Path('/dfs/share/maleiLab/wwt/eeg');out=root/'runs'/('full_T_recovery_audit_'+os.environ['SLURM_JOB_ID']);out.mkdir(mode=0o700);start=time.monotonic()
def write(name,value):
    p=out/name;t=p.with_suffix('.pending');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(p)
try:
    m=json.loads((root/'manifests/recovery_assets_20261108_v238.json').read_text());assert len(m['files'])==10
    for r in m['files']:
        p=root/r['relative_path'];assert p.resolve().parent==root/'data/full_T_recovery_20261108_v238'
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(1<<20),b''):h.update(b)
        assert h.hexdigest()==r['sha256'] and p.stat().st_size==r['bytes']
    import numpy as np
    data=root/'data/full_T_recovery_20261108_v238'
    split=json.loads((root/'data/recovery_20261108_v236/current_split_manifest.json').read_text())
    ids=json.loads((data/'T_identities.json').read_text());assert ids['subject']=='sub-01'
    rows=[r['array_row'] for r in ids['T']];assert rows==split['image_rows']['T'] and len(rows)==13230
    assert all(r['row']==i and r['partition']=='T' for i,r in enumerate(ids['T']))
    assert len({r['concept_id'] for r in ids['T']})==1323
    assert all(r['fold']==split['oof_fold_by_concept'][r['concept_id']] for r in ids['T'])
    with (data/'image_manifest.csv').open() as f:
        image={int(r['array_row']):r for r in csv.DictReader(f) if r['dataset_split']=='train'}
    assert len(image)==16540
    for row,r in zip(rows,ids['T']):
        assert all(image[row][k]==r[k] for k in ['concept_id','stimulus_id','image_sha256'])
    source=(data/'F_export_driver_original.py').read_text()
    assert "T=np.array(split['image_rows']['T'])" in source and "gallery=fused[T].clone().float().numpy()" in source
    assert "from eeg_labram_pretrained_20261028_v1 import build_encoder" in (data/'LaBraM_export_source_original.py').read_text()
    arrays={}
    for name in ['T_gallery_F.npy','T_raw_CLIP_H.npy','LaBraM_T_OOF_c.npy']:
        a=np.load(data/name,mmap_mode='r',allow_pickle=False);assert a.shape==(13230,1024) and a.dtype==np.float32
        assert np.isfinite(a).all();norms=np.linalg.norm(a,axis=1);assert (norms>0).all()
        arrays[name]=dict(shape=list(a.shape),dtype=str(a.dtype),norm_mean=float(norms.mean()),norm_min=float(norms.min()),norm_max=float(norms.max()))
    result=dict(status='COMPLETE_FULL_T_TARGET_AND_LABRAM_OOF_RECOVERY_AUDIT',job_id=os.environ['SLURM_JOB_ID'],hostname=socket.gethostname(),
        seconds=time.monotonic()-start,files_SHA_verified=10,rows=13230,concepts=1323,all_row_identity_joins_pass=True,arrays=arrays,
        F_is_original_HIVE_fusion_target=True,CLIP_is_separate_visual_target=True,OOF_condition_host='LaBraM',OOF_condition_subject='sub-01',
        raw_EEG_recovered_subject='sub-03',OOF_c_is_NOT_HIVE_BrainMLP_or_ATMS=True,parents_or_stochastic_teacher_training_inputs_restored=False,
        new_updates=0,new_images=0,new_efficacy=False)
    write('completion.json',result);print(json.dumps(result),flush=True)
except BaseException as exc:
    write('failure.json',dict(status='FAILED_PRESERVED',error=repr(exc),traceback=traceback.format_exc()));raise
