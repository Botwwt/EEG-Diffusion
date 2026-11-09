"""Independent area/metrics/bootstrap and sampled raw-EEG decoder replay.

Same NumPy backend; no independent full ridge refit, GPU or image encoder.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def checked(x):
    p=Path(x['path']);assert p.is_file() and sha(p)==x['sha256']
    if 'bytes' in x:assert p.stat().st_size==x['bytes']
    return p
def ref(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def rss():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
def run(spec_path):
    start=time.monotonic();rss_base=rss();spec=read(spec_path);B=Path(spec['output']);completion=read(B/'completion.json')
    assert completion['status']=='D38_COMPLETE13230_CONCEPT_HELD_ORDERED_AREA_DIAGNOSTIC'
    for x in spec['source_pins']:checked(x)
    checked(completion['spec']);pred=np.load(checked(completion['predictions']),mmap_mode='r',allow_pickle=False)
    archive=np.load(checked(completion['audit_arrays']),allow_pickle=False)
    D30=read(checked(spec['D30_spec']));physical=read(checked(D30['physical_R4_input_contract']))
    y=np.asarray(np.load(checked(physical['files']['targets.npy']),allow_pickle=False),np.float64)
    trials=np.load(checked(physical['files']['trials.npy']),mmap_mode='r',allow_pickle=False)
    index=[json.loads(s) for s in checked(physical['files']['index.jsonl']).read_text().splitlines()]
    raw=np.empty((13230,4250),np.float64)
    for low in range(0,len(raw),256):raw[low:low+256]=np.asarray(trials[low:low+256]).mean(1,dtype=np.float32).reshape(-1,4250)
    assert np.all(archive['coverage']==1) and np.array_equal(archive['folds'],[v['fold'] for v in index])
    area_error=forward_error=projected_error=decoder_error=normalizer_error=0.;vectors=0
    import torch
    torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    for fold in (0,1):
        data=np.load(B/f'fold{fold}_features.npz',allow_pickle=False)
        c,projected,area,scatter=data['native_conditions'],data['projected_increments'],data['area'],data['scatter']
        fit,held=data['fit_rows'],data['held_rows']
        assert set(fit)==set(np.flatnonzero(archive['folds']!=fold)) and set(held)==set(np.flatnonzero(archive['folds']==fold))
        basis,scales,mean=data['basis'],data['scales'],data['mean']
        assert np.allclose(basis.T@basis,np.eye(32),atol=2e-12)
        assert np.max(np.abs(projected[fit].mean(0)))<1e-11
        assert np.allclose(np.einsum('nti,ntj->ij',projected[fit],projected[fit])/(len(fit)*5),np.eye(32),atol=2e-10)
        assert 0<float(data['retained_variance'])<=1
        rows,cols=np.triu_indices(32,1)
        direct=np.zeros_like(area);unordered=np.zeros_like(scatter)
        for t in range(5):
            unordered+=projected[:,t,rows]*projected[:,t,cols]
            for u in range(t+1,5):
                direct+=.5*(projected[:,t,rows]*projected[:,u,cols]-projected[:,t,cols]*projected[:,u,rows])
        error=max(float(np.max(np.abs(area-direct))),float(np.max(np.abs(scatter-unordered))));area_error=max(area_error,error);assert error<1e-9
        # Raw EEG check uses separate NumPy64 native GELU/LN algebra and all original W_t.
        checkrows=np.concatenate((fit[np.linspace(0,len(fit)-1,64,dtype=int)],held[np.linspace(0,len(held)-1,64,dtype=int)]))
        state={k:v.numpy().astype(np.float64) for k,v in torch.load(checked(D30['roles']['oof'+str(fold)]['checkpoint']),weights_only=True,map_location='cpu')['model'].items()}
        W=state['proj_in.weight'];x=raw[checkrows];hidden=x@W.T+state['proj_in.bias']
        activation=hidden@state['block.layers.0.ff.w1.weight'].T
        erf=np.fromiter((math.erf(v/math.sqrt(2)) for v in activation.ravel()),dtype=np.float64,count=activation.size).reshape(activation.shape)
        z=hidden+(.5*activation*(1+erf))@state['block.layers.0.ff.w2.weight'].T
        z=z-z.mean(1,keepdims=True);z=z/np.sqrt(np.square(z).mean(1,keepdims=True)+1e-6)
        truth=z*state['block.layers.0.norm.weight']+state['block.layers.0.norm.bias']
        error=float(np.max(np.abs(truth-c[checkrows])));forward_error=max(forward_error,error);assert error<3e-5
        w=W.reshape(1024,17,250);xx=x.reshape(-1,17,250)
        parts=np.stack([xx[:,:,lo:hi].reshape(len(x),-1)@w[:,:,lo:hi].reshape(1024,-1).T for lo,hi in zip((0,50,100,150,200),(50,100,150,200,250))],axis=1)
        truth_projection=((parts-mean)@basis)/scales
        error=float(np.max(np.abs(truth_projection-projected[checkrows])));projected_error=max(projected_error,error);assert error<3e-5
        common=np.concatenate((c,raw,scatter),axis=1)
        common_fit=common[fit];mu=common_fit.mean(0);sd=np.sqrt(np.square(common_fit-mu).mean(0));sd[sd<1e-10]=1.
        chosen=checkrows[64:]
        for arm,name in enumerate(completion['arms']):
            record=next(v for v in completion['model_records'] if v['held_fold']==fold and v['arm']==name)
            model=np.load(checked(record['model']),allow_pickle=False);dim=model['common_coefficient'].shape[0]
            error=max(float(np.max(np.abs(model['input_mean']-mu[:dim]))),float(np.max(np.abs(model['input_scale']-sd[:dim]))));normalizer_error=max(normalizer_error,error);assert error<1e-11
            predicted=((common[chosen,:dim]-model['input_mean'])/model['input_scale'])@model['common_coefficient']+model['target_mean']
            if arm>=3:
                extra=area if arm==3 else area*archive['random_signs'][:,None] if arm==4 else area[archive['wrong_rows']]
                m=extra[fit].mean(0);s=np.sqrt(np.square(extra[fit]-m).mean(0));s[s<1e-10]=1.
                assert np.allclose(model['area_mean'],m,atol=1e-11) and np.allclose(model['area_scale'],s,atol=1e-11)
                predicted+=((extra[chosen]-m)/s)@model['area_coefficient']
            error=float(np.max(np.abs(predicted-pred[chosen,arm])));decoder_error=max(decoder_error,error);assert error<3e-6
            vectors+=len(chosen)
        del common,common_fit,c,projected,area,scatter,direct,unordered,data
        assert time.monotonic()-start<840 and rss()-rss_base<10<<30
    yn=y/np.linalg.norm(y,axis=1,keepdims=True);values=np.empty((13230,6,5))
    for lo in range(0,13230,128):
        end=min(lo+128,13230);p=np.asarray(pred[lo:end],np.float64)
        pn=p/np.maximum(np.linalg.norm(p,axis=-1,keepdims=True),1e-30)
        sim=np.einsum('nad,nkd->nak',pn,yn[archive['candidates'][lo:end]])
        selected=np.take_along_axis(sim,archive['labels'][lo:end,None,None].repeat(6,axis=1),axis=-1)[...,0]
        maxima=sim.max(-1)
        top=(selected==maxima)/(sim==maxima[...,None]).sum(-1)
        afc=((selected[...,None]>sim).sum(-1)+.5*(selected[...,None]==sim).sum(-1)-.5)/9
        values[lo:end]=np.stack((np.square(p-y[lo:end,None]).mean(-1),selected,top,afc,selected-(sim.sum(-1)-selected)/9),axis=-1)
    metric_error=float(np.max(np.abs(values-archive['metrics'])));assert metric_error<1e-12
    groups={}
    for i,v in enumerate(index):groups.setdefault(v['concept_id'],[]).append(i)
    concept=np.stack([values[p].mean(0) for p in groups.values()]);assert np.allclose(concept,archive['concept_metrics'],atol=1e-13)
    indices=np.random.default_rng(1729).integers(0,1323,size=(1000,1323));assert np.array_equal(indices,archive['bootstrap_indices'])
    draws=np.concatenate([concept[indices[i:i+20]].mean(1) for i in range(0,1000,20)])
    assert np.allclose(draws,archive['bootstrap_means'],atol=1e-13)
    for control in (1,2,4,5):
        difference=concept[:,3]-concept[:,control];difference[:,0]*=-1
        boot=np.concatenate([difference[indices[i:i+20]].mean(1) for i in range(0,1000,20)])
        lo,hi=np.quantile(boot,[.025,.975],axis=0);v=completion['ordered_area_minus_controls'][completion['arms'][control]]
        assert np.allclose(difference.mean(0),v['mean'],atol=1e-13)
        assert np.allclose(lo,v['descriptive95_lower'],atol=1e-13) and np.allclose(hi,v['descriptive95_upper'],atol=1e-13)
    comp=completion['ordered_area_minus_controls'];main=comp['c_raw_scatter']
    gate=(main['relative_MSE_improvement']>=.005 and main['mean'][2]>=.01 and all(v['descriptive95_lower'][0]>0 and v['mean'][2]>=0 for v in comp.values()))
    assert gate==completion['decision']['predictive_gate_passed']
    assert not torch.cuda.is_initialized()
    result=dict(status='D38_INDEPENDENT_AREA_ALL_METRICS_BOOTSTRAPS_AND_SAMPLED_RAW_DECODERS_CPU_PASSED',
       UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=ref(Path(__file__).resolve()),spec=ref(spec_path),
       primary_completion=ref(B/'completion.json'),all_area_and_scatter_rows=26460,all_scalar_metrics=values.size,
       all4_fixed_contrasts_1000_bootstraps_verified=True,sampled_raw_EEG_rows_per_role=128,
       sampled_held_prediction_vectors=vectors,maximum_area_error=area_error,maximum_native_NumPy64_forward_error=forward_error,
       maximum_projected_NumPy64_error=projected_error,maximum_saved_decoder_prediction_error=decoder_error,
       maximum_FIT_normalizer_error=normalizer_error,maximum_scalar_metric_error=metric_error,
       independent_full_ridge_refit_or_all_raw_feature_forward=False,same_NumPy_backend=True,
       no_GPU_or_new_neural_fit_or_images=True,seconds=time.monotonic()-start,peak_extra_CPU_bytes=rss()-rss_base)
    with (B/'independent_CPU_audit.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);run(p.parse_args().spec)
