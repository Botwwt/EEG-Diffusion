"""Slurm CPU-only audit of recovered real EEG and checkpoints. No fitting."""
import hashlib,json,os,socket,time,traceback
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
root=Path('/dfs/share/maleiLab/wwt/eeg');start=time.monotonic()
out=root/'runs'/('recovery_audit_'+os.environ['SLURM_JOB_ID']);out.mkdir(mode=0o700)
def write(name,value):
    p=out/name;t=p.with_suffix('.pending');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(p)
try:
    manifest=json.loads((root/'manifests/recovery_assets_20261108_v236.json').read_text())
    assert manifest['status']=='BACKUPS_TRANSFERRED_AWAITING_COMPUTE_SHA_AND_SCHEMA'
    checks=[]
    for r in manifest['files']:
        p=root/r['relative_path'];assert p.resolve().parent in [root/'data/recovery_20261108_v236',root/'checkpoints/recovery_20261108_v236',root/'code']
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(1<<20),b''):h.update(b)
        assert p.stat().st_size==r['bytes'] and h.hexdigest()==r['sha256'],r['relative_path']
        checks.append(dict(path=r['relative_path'],bytes=r['bytes'],sha256=r['sha256']))
    write('hash_verification.json',dict(status='ALL_RECOVERED_FILES_SHA256_PASS',files=checks))
    import numpy as np
    import torch
    torch.set_num_threads(4)
    data=root/'data/recovery_20261108_v236'
    eeg=torch.load(data/'sub-03_train_mean4.pt',map_location='cpu',weights_only=True)
    layout=json.loads((data/'sub-03_layout.json').read_text())
    assert isinstance(eeg,torch.Tensor) and eeg.shape==(16540,63,250) and eeg.dtype==torch.float32
    assert layout['subject']=='sub-03' and layout['raw_shape']==[16540,4,63,250]
    assert layout['shape']==list(eeg.shape) and layout['selected_image_indices']==list(range(16540))
    assert layout['input_is_already_author_MVNN'] and len(layout['channels'])==63 and len(set(layout['channels']))==63
    extrema=[]
    for i in range(0,len(eeg),256):
        block=eeg[i:i+256];assert torch.isfinite(block).all()
        extrema.append([float(block.min()),float(block.max())])
    bank=np.load(data/'fit_bank_unit.npy',mmap_mode='r',allow_pickle=False)
    contract=json.loads((data/'data_contract.json').read_text())
    ids=contract['fit_bank_original_ids']
    assert bank.shape==(11630,1024) and bank.dtype==np.float32 and len(ids)==11630
    assert len(set(ids))==len(ids) and ids==sorted(ids) and all(0<=i<16540 for i in ids)
    assert np.isfinite(bank).all()
    norms=np.linalg.norm(bank,axis=1);assert np.max(np.abs(norms-1))<2e-6
    split=json.loads((data/'current_split_manifest.json').read_text());assert split['training_seed']==1729
    permitted={k:set(v) for k,v in split['image_rows'].items()}
    assert not permitted['T']&permitted['D'] and not permitted['T']&permitted['V']
    coverage={k:len(set(ids)&v) for k,v in permitted.items()}
    assert sum(coverage.values())==len(ids)
    shared_T=sorted(set(ids)&permitted['T']);assert shared_T
    # Export actual T-only raw measurements for a later, separately admitted host probe.
    sample_ids=shared_T[:8];rows=torch.tensor(sample_ids)
    torch.save(dict(eeg_R4=eeg[rows].clone(),array_rows=rows,subject='sub-03',channels=layout['channels']),out/'real_T8_EEG.pt')
    ordinary=torch.load(root/'checkpoints/recovery_20261108_v236/ordinary_full1654.pt',map_location='cpu',weights_only=True)
    def describe(value,depth=0):
        if isinstance(value,torch.Tensor):return dict(shape=list(value.shape),dtype=str(value.dtype),finite=bool(torch.isfinite(value).all()))
        if isinstance(value,dict) and depth<3:return {str(k):describe(v,depth+1) for k,v in value.items()}
        if isinstance(value,(str,int,float,bool,type(None))):return value
        return dict(type=type(value).__name__)
    write('ordinary_checkpoint_schema.json',describe(ordinary))
    result=dict(status='COMPLETE_REAL_ASSET_RECOVERY_AUDIT_ONLY',job_id=os.environ['SLURM_JOB_ID'],hostname=socket.gethostname(),
        seconds=time.monotonic()-start,files_verified=len(checks),EEG_shape=list(eeg.shape),EEG_subject='sub-03',EEG_repeats=4,
        EEG_min=min(v[0] for v in extrema),EEG_max=max(v[1] for v in extrema),visual_bank_shape=list(bank.shape),
        visual_bank_current_split_coverage=coverage,T8_array_rows=sample_ids,
        alignment_scope='Published array-order identities, not independent event replay',
        preproc_train_only_independently_established=False,visual_bank_is_HIVE_Fusion_F=False,
        exact_latest_old_trainer_resumed=False,new_training_updates=0,new_images=0,new_efficacy_established=False)
    write('completion.json',result);print(json.dumps(result),flush=True)
except BaseException as exc:
    write('failure.json',dict(status='FAILED_PRESERVED',error=repr(exc),traceback=traceback.format_exc()))
    raise
