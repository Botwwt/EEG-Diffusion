"""Export all eight EEG-only condition arms after all three fixed-final fits."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
import argparse,datetime,hashlib,json,math,time,zipfile
from pathlib import Path
import numpy as np
import torch
import brainhive_spatial_images_20261028_v1 as U

def gelu_np(x):
    e=np.fromiter((math.erf(v/math.sqrt(2)) for v in x.ravel()),dtype=np.float64,count=x.size).reshape(x.shape)
    return .5*x*(1+e)
def native_np(x,s):
    h=x@s['proj_in.weight'].T+s['proj_in.bias']
    z=h+gelu_np(h@s['block.layers.0.ff.w1.weight'].T)@s['block.layers.0.ff.w2.weight'].T
    z-=z.mean(-1,keepdims=True);z/=np.sqrt(np.square(z).mean(-1,keepdims=True)+1e-6)
    return z*s['block.layers.0.norm.weight']+s['block.layers.0.norm.bias']
def ordinary_np(c,gamma,beta,s):
    u=(c-beta)/gamma;h=gelu_np(u@s['trunk.0.weight'].T+s['trunk.0.bias'])
    d=h@s['direction.weight'].T+s['direction.bias'];d-=d.mean(-1,keepdims=True)
    r2=np.square(u).sum(-1,keepdims=True);d-=(d*u).sum(-1,keepdims=True)/r2*u
    delta=.25*np.sqrt(r2)*d/np.sqrt(r2+np.square(d).sum(-1,keepdims=True))
    scale=np.sqrt(r2)/np.linalg.norm(u+delta,axis=-1,keepdims=True)
    return c+gamma*((scale-1)*u+scale*delta)

def run(path):
    started=time.monotonic();torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    execution,policy,train,base,queries=U.load_execution(path);_,fits=U.complete_fits(execution)
    out=U.EVAL/'prepared';assert not out.exists();out.mkdir(parents=True)
    mapping=U.read(U.checked(base['EEG_mapping']));assert mapping['Dcheck_queries']==queries
    spatial=U.read(U.checked(train['spatial_CPU']));sf=spatial['files']
    eeg=np.load(U.checked(sf['Dcheck_mean63']),allow_pickle=False)
    assert eeg.shape==(400,63,250) and eeg.dtype==np.float32 and np.isfinite(eeg).all()
    import brainhive_paired_study_contract_20261025_v1 as C
    eng=U.read(U.checked(train['engineering']));es=U.read(U.checked(eng['spec']))
    C.setup_runtime(dict(runtime_manifest=es['runtime_manifest']))
    from brainhive_reversible_eeg_20261028_v1 import author_brain_class
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    cpu=U.read(U.checked(es['CPU_completion']));cs=U.read(U.checked(cpu['spec']));d30=U.read(U.checked(cs['D30_spec']))
    cls=author_brain_class(U.checked(d30['author_models_source']),U.checked(d30['author_layers_source']))
    parent=torch.load(U.checked(base['parent_checkpoint']),map_location='cpu',weights_only=True)['model']
    frozen=U.read(U.checked(es['ordinary_spec']));old=torch.load(U.checked(frozen['checkpoint']),map_location='cpu',weights_only=True)
    assert frozen['checkpoint']['sha256']==policy['old_ordinary_sha256'] and old['calibration']['selected_strength']==1
    ordinary=BoundedConditionAdapter(predict_risk=False).eval().requires_grad_(False);ordinary.load_state_dict(old['model'],strict=True)
    gamma=parent['block.layers.0.norm.weight'];beta=parent['block.layers.0.norm.bias']
    result=np.empty((400,8,1024),np.float32);maxima={}
    for e in U.ENCODERS:
        import brainhive_spatial_innovation_20261028_v1 as M
        original=cls(4250,1024,num_layers=1,dropout=.1).eval().requires_grad_(False);original.load_state_dict(parent,strict=True)
        if e=='original':model=original;state=parent
        else:
            stats={n:torch.from_numpy(np.load(U.checked(sf[n]),allow_pickle=False)) for n in ['L','conditional_white','marginal_white','temporal_white','weight_Frobenius_radius']}
            mean=torch.from_numpy(np.load(U.checked(sf['mean']),allow_pickle=False))
            learned=torch.load(U.checked(fits[e]),map_location='cpu',weights_only=True);assert set(learned['model'])=={'B'}
            model=M.SpatialNative(original,stats,mean,e).eval().requires_grad_(False);model.B.copy_(learned['model']['B'])
            state=model.folded_state()
        n=U.ARMS.index(e+'__native');a=U.ARMS.index(e+'__ordinary')
        with torch.no_grad():
            for lo in range(0,400,16):
                x=torch.from_numpy(eeg[lo:lo+16]);c=model(x[:,46:].flatten(1)) if e=='original' else model(x)
                o=ordinary(c,gamma.expand_as(c),beta.expand_as(c),gate_mode='fixed')[0]
                result[lo:lo+16,n]=c.numpy();result[lo:lo+16,a]=o.numpy()
        # Independently reconstruct every raw EEG output and old ordinary in
        # NumPy64. No cached predicted condition substitutes for these outputs.
        s={k:v.numpy().astype(np.float64) for k,v in state.items()}
        os_={k:v.numpy().astype(np.float64) for k,v in old['model'].items()}
        max_native=max_ordinary=max_relative=0.
        for lo in range(0,400,32):
            x=(eeg[lo:lo+32,46:] if e=='original' else eeg[lo:lo+32]).reshape(len(eeg[lo:lo+32]),-1).astype(np.float64);c=native_np(x,s)
            o=ordinary_np(c,gamma.numpy().astype(np.float64),beta.numpy().astype(np.float64),os_)
            max_native=max(max_native,float(np.abs(c-result[lo:lo+32,n]).max()))
            max_ordinary=max(max_ordinary,float(np.abs(o-result[lo:lo+32,a]).max()))
            max_relative=max(max_relative,float((np.linalg.norm(o-result[lo:lo+32,a],axis=1)/np.linalg.norm(o,axis=1)).max()))
        assert max_native<1e-3 and max_ordinary<1e-3 and max_relative<1e-4
        maxima[e]=dict(native_NumPy64_max_error=max_native,ordinary_NumPy64_max_error=max_ordinary,ordinary_relative_L2_max_error=max_relative)
        assert time.monotonic()-started<550
    assert np.isfinite(result).all() and not torch.cuda.is_initialized()
    historical=np.load(U.checked(base['baseline_conditions']),allow_pickle=False)
    drift=dict(native_max=float(np.abs(result[:,0]-historical[:,0]).max()),ordinary_max=float(np.abs(result[:,1]-historical[:,1]).max()))
    assert drift['native_max']<1e-3 and drift['ordinary_max']<1e-3
    bundle=U.read(U.checked(execution['target_bundle_manifest']));target_rows=[]
    assert bundle['status']=='D39_ORIGINAL_D09_ALL400_TARGET_JPEGS_AND_IDENTITY_BOUND'
    assert bundle['original_feature_cache']['sha256']==base['target_features']['sha256']
    with zipfile.ZipFile(U.checked(execution['target_bundle'])) as archive:
        assert json.loads(archive.read('bundle_manifest.json'))==bundle
        for name,item in [('original_target_metadata.json',bundle['metadata']),('original_D09_inputs.json',bundle['D09_input_completion']),('original_D09_feature_completion.json',bundle['original_feature_completion'])]:
            payload=archive.read(name);assert hashlib.sha256(payload).hexdigest()==item['sha256']
        metadata=json.loads(archive.read('original_target_metadata.json'))['Dcheck']
        inputs=json.loads(archive.read('original_D09_inputs.json'));encoded=json.loads(archive.read('original_D09_feature_completion.json'))
        assert inputs['target_metadata']['sha256']==bundle['metadata']['sha256'] and encoded['inputs']['sha256']==bundle['D09_input_completion']['sha256']
        assert encoded['check_target_features']['sha256']==base['target_features']['sha256']
        assert len(bundle['targets'])==len(metadata)==400
        for i,(q,row,original) in enumerate(zip(queries,bundle['targets'],metadata)):
            assert all(q[k]==row[k]==original[k] for k in U.IDENTITY)
            data=archive.read(row['name']);assert hashlib.sha256(data).hexdigest()==row['sha256']==original['image_sha256']
            assert len(data)==row['bytes']==int(original['image_bytes'])
            dest=U.EVAL/'targets'/f'{i:03d}.jpg';dest.parent.mkdir(exist_ok=True)
            with dest.open('xb') as f:f.write(data)
            target_rows.append(dict(**q,target_index=i,image=U.ref(dest),original_source_archive=original['image_zip'],original_source_member=original['image_member']))
    U.write(out/'target_rows.json',target_rows)
    U.array(out/'conditions400x8.npy',result)
    U.write(out/'queries.json',queries);U.evaluation_budget()
    v=dict(status='D40_ALL400x8_FRESH_NATIVE_CORE_CONDITIONS_CPU_EXPORTED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      execution=U.ref(path),training_queue_completion=U.ref(U.B/'training_queue_v1/completion.json'),models=dict(original=base['parent_checkpoint'],**fits),
      ordinary_checkpoint=frozen['checkpoint'],ordinary_strength=1,ordinary_affine=base['parent_checkpoint'],
      arms=list(U.ARMS),query_ids=[q['query_id'] for q in queries],queries=U.ref(out/'queries.json'),conditions=U.ref(out/'conditions400x8.npy'),
      EEG=sf['Dcheck_mean63'],EEG_mapping=base['EEG_mapping'],CPU_export_batch_size=16,
      target_rows=U.ref(out/'target_rows.json'),target_bundle=execution['target_bundle'],original_target_bytes_and_D09_source_chain_verified=True,
      native_and_ordinary_NumPy64_all400x8_replay=maxima,historical_cached_condition_drift=drift,
      same_NumPy_backend_not_an_independent_fit=True,fresh_original_conditions_not_replaced_by_historical_cache=True,
      target_features_or_reconstruction_scores_read=False,new_GPU_jobs=0,new_model_fits=0,seconds=time.monotonic()-started)
    U.write(out/'completion.json',v);print(__import__('json').dumps(v),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execution',type=Path,required=True);run(p.parse_args().execution)
