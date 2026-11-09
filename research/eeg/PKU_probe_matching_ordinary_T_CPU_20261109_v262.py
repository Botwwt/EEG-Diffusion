"""Read-only all-T geometry of a coherent live checkpoint; never select epochs."""
import hashlib,io,json,os,socket,time,traceback
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
R=Path('/dfs/share/maleiLab/wwt/eeg');out=R/('runs/ordinary_T_geometry_'+os.environ['SLURM_JOB_ID']);out.mkdir();start=time.monotonic()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def write(n,v):
    b=(json.dumps(v,indent=2,allow_nan=False)+'\n').encode();p=out/n;t=p.with_name(p.name+'.pending')
    with t.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    assert t.read_bytes()==b;t.replace(p)
try:
    import numpy as np
    import torch
    from PKU_ordinary_component_source_v253 import BoundedConditionAdapter
    torch.set_num_threads(4)
    parent=R/'runs/HIVE_native_sub03_5096542';fit=R/'runs/HIVE_matching_ordinary_sub03_v259'
    # One open/read sees the old or new complete inode across atomic replacement.
    raw=(fit/'resume.pt').read_bytes();digest=hashlib.sha256(raw).hexdigest();snapshot=out/'checkpoint_snapshot.pt'
    with snapshot.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    assert sha(snapshot)==digest;state=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False);del raw
    assert state['input_provenance_sha256']==sha(fit/'input_provenance.json')
    bank=torch.load(parent/'T_true_OOF_conditions.pt',map_location='cpu',weights_only=True);ids=bank['array_rows'].tolist();assert len(ids)==13230
    identity=json.loads((R/'data/full_T_recovery_20261108_v238/T_identities.json').read_text());assert [r['array_row'] for r in identity['T']]==ids
    concepts=[r['concept_id'] for r in identity['T']];mapping={c:i for i,c in enumerate(sorted(set(concepts)))};codes=np.array([mapping[c] for c in concepts])
    assert len(mapping)==1323 and (np.bincount(codes)==10).all()
    split=json.loads((R/'data/recovery_20261108_v236/current_split_manifest.json').read_text());assert ids==split['image_rows']['T']
    fullF=torch.load(parent/'deterministic_F_all16540.pt',map_location='cpu',weights_only=True);target=fullF[ids].double().numpy();del fullF
    target/=np.linalg.norm(target,axis=1,keepdims=True)
    model=BoundedConditionAdapter(predict_risk=False).eval();model.load_state_dict(state['model'],strict=True)
    affine=[]
    for role in ['oof0','oof1']:
        p=torch.load(parent/role/'brain_final.pt',map_location='cpu',weights_only=True)['model'];affine.append((p['block.layers.0.norm.weight'],p['block.layers.0.norm.bias']))
    weight=torch.stack([x[0] for x in affine]);bias=torch.stack([x[1] for x in affine]);c=bank['condition'];folds=bank['folds'].long()
    with torch.no_grad():changed=torch.cat([model(c[i:i+256],weight[folds[i:i+256]],bias[folds[i:i+256]],gate_mode='fixed')[0] for i in range(0,13230,256)])
    results={};arrays={};ever=state['coverage'].any(0).numpy();assert int(state['coverage'].sum())==state['seen']
    for arm,value in [('native',c),('ordinary',changed)]:
        x=value.double().numpy();assert np.isfinite(x).all();energy=float(np.sum(np.mean(x,axis=0)**2)/np.mean(np.sum(x*x,axis=1)))
        x=x/np.linalg.norm(x,axis=1,keepdims=True);own=[];within=[];global_margin=[];global_top=[];within_top=[]
        for lo in range(0,13230,128):
            hi=min(lo+128,13230);ix=np.arange(lo,hi);scores=x[lo:hi]@target.T;positive=scores[np.arange(len(ix)),ix].copy()
            same=codes[ix,None]==codes[None,:];within_top.extend((np.where(same,scores,-np.inf).argmax(1)==ix).tolist());global_top.extend((scores.argmax(1)==ix).tolist())
            scores[np.arange(len(ix)),ix]=-np.inf;within.extend((positive-np.where(same,scores,-np.inf).max(1)).tolist());global_margin.extend((positive-scores.max(1)).tolist());own.extend(positive.tolist())
        own=np.asarray(own);within=np.asarray(within);global_margin=np.asarray(global_margin);global_top=np.asarray(global_top);within_top=np.asarray(within_top)
        for n,v in [('own_cosine',own),('within_margin',within),('global_margin',global_margin),('within_top1',within_top),('global_top1',global_top)]:arrays[arm+'_'+n]=v
        def summary(mask):return dict(queries=int(mask.sum()),own_cosine=float(own[mask].mean()),within_margin=float(within[mask].mean()),global_margin=float(global_margin[mask].mean()),
            within_top1=float(within_top[mask].mean()),global_top1=float(global_top[mask].mean()),strict_within_top1=float((within[mask]>0).mean()),strict_global_top1=float((global_margin[mask]>0).mean()))
        results[arm]=dict(allT=summary(np.ones(13230,dtype=bool)),raw_condition_mean_energy_fraction=energy,exposed_T=summary(ever) if ever.any() else None,
            not_yet_exposed_T=summary(~ever) if (~ever).any() else None)
    delta={}
    for name in ['own_cosine','within_margin','global_margin','within_top1','global_top1']:
        a=arrays['native_'+name];b=arrays['ordinary_'+name];d=b.astype(np.float64)-a.astype(np.float64)
        delta[name]=dict(mean=float(d.mean()),minimum=float(d.min()),maximum=float(d.max()),worse_queries=int((d<0).sum()),better_queries=int((d>0).sum()))
    np.savez_compressed(out/'all13230_paired_feature_metrics.npz',array_rows=np.asarray(ids),concept_codes=codes,ever_exposed=ever,**arrays)
    result=dict(status='COMPLETE_ALL13230_MATCHED_ORDINARY_T_GEOMETRY_READONLY_DIAGNOSTIC',job_id=os.environ['SLURM_JOB_ID'],host=socket.gethostname(),seconds=time.monotonic()-start,
        captured_training_updates=state['updates'],captured_exposures=state['seen'],ever_exposed_T=int(ever.sum()),checkpoint_sha256=digest,
        results=results,paired_difference=delta,metric_table_sha256=sha(out/'all13230_paired_feature_metrics.npz'),FP64_full13230_gallery=True,
        interpretation='Upstream EEG conditions are OOF; ordinary trains on T. This is a training diagnostic, not independent confirmation, image gain or a rule for epoch selection.',
        no_training_or_control_changes=True,new_updates=0,new_images=0,D_V_or_official_test_forward=False)
    assert sum(p.stat().st_size for p in out.iterdir() if p.is_file())<32<<20
    write('completion.json',result);os.chmod(snapshot,0o400);print(json.dumps(result),flush=True)
except BaseException as e:
    write('failure.json',dict(status='FAILED_PRESERVED',error=repr(e),traceback=traceback.format_exc()));raise
