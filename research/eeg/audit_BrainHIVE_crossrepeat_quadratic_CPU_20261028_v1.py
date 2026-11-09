"""Independent NumPy metrics, all FIT normalizers, and forty forward probes.

This does not independently refit the ten ridge models or replay every network
prediction. All saved13230x5 scores and1000 bootstraps are independently read.
"""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='2'
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
import traceback
import numpy as np

START=time.monotonic()


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_bytes())


def check_ref(item):
    path=Path(item['path']);assert path.is_file() and sha(path)==item['sha256']
    if 'bytes' in item:assert path.stat().st_size==item['bytes']
    return path


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())


def rss():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024


BASE=rss()


def budget():
    assert time.monotonic()-START<280 and rss()-BASE<10<<30
    assert 'torch' not in sys.modules and 'diffusers' not in sys.modules


def quadratic(raw,mu,scale,A,B):
    x=(np.asarray(raw,np.float64).reshape(len(raw),4,4250)-mu)/scale
    a=(x.reshape(-1,4250)@A).reshape(len(raw),4,128)
    b=(x.reshape(-1,4250)@B).reshape(len(raw),4,128)
    naive=a.mean(1)*b.mean(1)
    direct=np.zeros_like(naive)
    for left in range(4):
        for right in range(4):
            if left!=right:direct+=a[:,left]*b[:,right]/12
    centered=np.zeros_like(naive)
    for repeat in range(4):centered+=(a[:,repeat]-a.mean(1))*(b[:,repeat]-b.mean(1))/12
    assert np.allclose(direct,naive-centered,atol=1e-10,rtol=1e-10)
    return naive,direct,centered


def run(spec_path,completion,output):
    assert not output.exists();output.mkdir(parents=True)
    spec=read(spec_path);done=read(completion)
    assert done['status']=='D32_COMPLETE_FULL_T_CROSS_REPEAT_QUADRATIC_PREDICTIVE_DIAGNOSTIC'
    assert done['spec_sha256']==sha(spec_path)
    for item in spec['source_pins']:check_ref(item)
    assert done['analytical_models']==10 and done['new_GPU_jobs']==done['new_SGD_updates']==done['new_images']==0
    upstream=read(check_ref(spec['upstream_D30_spec']));physical=read(check_ref(upstream['physical_R4_input_contract']));refs=physical['files']
    for name in ('trials.npy','targets.npy','index.jsonl'):check_ref(refs[name])
    trials=np.load(refs['trials.npy']['path'],mmap_mode='r',allow_pickle=False)
    target=np.load(refs['targets.npy']['path'],mmap_mode='r',allow_pickle=False).astype(np.float64)
    prediction=np.load(check_ref(done['predictions']),mmap_mode='r',allow_pickle=False)
    arrays=np.load(check_ref(done['audit_arrays']),allow_pickle=False)
    index=[json.loads(line) for line in Path(refs['index.jsonl']['path']).read_text().splitlines()]
    split=read(check_ref(spec['split']));groups={}
    for row in index:groups.setdefault(row['concept_id'],[]).append(row['row'])
    assert len(index)==13230 and len(groups)==1323 and all(len(v)==10 for v in groups.values())
    assert np.all(arrays['coverage']==1)
    folds=np.array([split['oof_fold_by_concept'][row['concept_id']] for row in index],np.int8)
    assert np.array_equal(folds,arrays['folds'])
    metric=np.empty((13230,5,5),np.float64)
    for positions in groups.values():
        p=np.asarray(prediction[positions],np.float64);t=target[positions]
        expected=np.broadcast_to(positions,(10,10))
        assert np.array_equal(arrays['candidate_rows'][positions],expected) and np.array_equal(arrays['labels'][positions],np.arange(10))
        assert np.array_equal(arrays['wrong_U_rows'][positions],np.roll(positions,-1))
        for method in range(5):
            unit=p[:,method]/np.maximum(np.linalg.norm(p[:,method],axis=1,keepdims=True),1e-30)
            ty=t/np.linalg.norm(t,axis=1,keepdims=True)
            similarities=unit@ty.T
            for local,row in enumerate(positions):
                correct=similarities[local,local];wrong=np.delete(similarities[local],local)
                maximum=similarities[local].max();tie_count=np.count_nonzero(similarities[local]==maximum)
                top=float(correct==maximum)/tie_count
                two=float(np.count_nonzero(correct>wrong)+.5*np.count_nonzero(correct==wrong))/9
                metric[row,method]=[np.square(p[local,method]-t[local]).mean(),correct,top,two,correct-wrong.mean()]
        budget()
    error=float(np.abs(metric-arrays['metrics']).max());assert error<1e-10
    concept=np.stack([metric[rows].mean(0) for rows in groups.values()])
    assert np.allclose(concept,arrays['concept_metrics'],atol=1e-10,rtol=1e-10)
    bootstrap=np.random.default_rng(1729).integers(0,1323,size=(1000,1323))
    assert np.array_equal(bootstrap,arrays['bootstrap_indices'])
    draws=np.array([concept[chosen].mean(0) for chosen in bootstrap])
    assert np.allclose(draws,arrays['bootstrap_means'],atol=1e-10,rtol=1e-10)
    for control in (0,1,3,4):
        name=done['arms'][control];value=done['U_minus_controls'][name]
        delta=concept[:,2]-concept[:,control];delta[:,0]*=-1
        sampled=np.array([delta[chosen].mean(0) for chosen in bootstrap]);low,high=np.quantile(sampled,[.025,.975],axis=0)
        assert np.allclose(delta.mean(0),value['mean'],atol=1e-10,rtol=1e-10)
        assert np.allclose(low,value['descriptive_95_lower'],atol=1e-10,rtol=1e-10) and np.allclose(high,value['descriptive_95_upper'],atol=1e-10,rtol=1e-10)
        relative=float((concept[:,control,0].mean()-concept[:,2,0].mean())/concept[:,control,0].mean())
        assert abs(relative-value['relative_MSE_improvement'])<1e-10
    comparisons=done['U_minus_controls'];linear=comparisons['linear_mean4'];naive=comparisons['mean4_naive_quadratic']
    supports=(linear['relative_MSE_improvement']>=.005 and naive['relative_MSE_improvement']>=.005
        and linear['mean'][2]>=.01 and naive['mean'][2]>=0 and all(x['descriptive_95_lower'][0]>0 for x in comparisons.values()))
    assert supports==done['practical_predictive_gate_passed']
    # All FIT normalizers are recalculated from physical records. Full models
    # are independently forwarded on20 fixed evenly-spaced held rows per fold.
    mean=np.empty((13230,4250),np.float64)
    for lo in range(0,13230,256):mean[lo:lo+256]=np.asarray(trials[lo:lo+256],np.float64).mean(1).reshape(-1,4250);budget()
    generator=np.random.default_rng(1729)
    A=(generator.integers(0,2,size=(4250,128),dtype=np.int8)*2-1).astype(np.float64)/np.sqrt(4250.)
    B=(generator.integers(0,2,size=(4250,128),dtype=np.int8)*2-1).astype(np.float64)/np.sqrt(4250.);B[:,:64]=A[:,:64]
    forward_error=normalizer_error=0.;checks=[]
    for fold in (0,1):
        fit=np.flatnonzero(folds!=fold);held=np.flatnonzero(folds==fold)
        selected=held[np.linspace(0,len(held)-1,20,dtype=np.int64)]
        mu=mean[fit].mean(0);scale=np.sqrt(np.square(mean[fit]-mu).mean(0));scale[scale<1e-10]=1
        qfit=[np.empty((len(fit),128),np.float64) for _ in range(3)]
        for lo in range(0,len(fit),128):
            batch=fit[lo:lo+128];values=quadratic(trials[batch],mu,scale,A,B)
            for v,result in zip(values,qfit):result[lo:lo+len(batch)]=v
            budget()
        qselected=quadratic(trials[selected],mu,scale,A,B)
        wrong_rows=arrays['wrong_U_rows'][selected]
        wrong_selected=quadratic(trials[wrong_rows],mu,scale,A,B)[1]
        lookup={int(row):position for position,row in enumerate(fit)}
        wrong_fit=qfit[1][[lookup[int(arrays['wrong_U_rows'][row])] for row in fit]]
        features_fit=[None,qfit[0],qfit[1],qfit[2],wrong_fit]
        features_selected=[None,qselected[0],qselected[1],qselected[2],wrong_selected]
        for arm,name in enumerate(done['arms']):
            record=next(x for x in done['model_records'] if x['held_fold']==fold and x['arm']==name)
            path=check_ref(record['model']);model=np.load(path,allow_pickle=False)
            assert model['linear'].shape==(4250,1024) and float(model['penalty'])==len(fit)*.01
            assert np.array_equal(model['A'],A) and np.array_equal(model['B'],B)
            assert np.allclose(model['input_mean'],mu,atol=1e-10,rtol=1e-10) and np.allclose(model['input_scale'],scale,atol=1e-10,rtol=1e-10)
            assert np.allclose(model['target_mean'],target[fit].mean(0),atol=1e-12,rtol=1e-12)
            actual=((mean[selected]-mu)/scale)@model['linear']+model['target_mean']
            if arm:
                qmean=features_fit[arm].mean(0);qscale=np.sqrt(np.square(features_fit[arm]-qmean).mean(0));qscale[qscale<1e-10]=1
                assert np.allclose(qmean,model['quadratic_mean'],atol=1e-10,rtol=1e-10)
                assert np.allclose(qscale,model['quadratic_scale'],atol=1e-10,rtol=1e-10)
                normalizer_error=max(normalizer_error,float(np.abs(qmean-model['quadratic_mean']).max()),float(np.abs(qscale-model['quadratic_scale']).max()))
                actual+=((features_selected[arm]-qmean)/qscale)@model['quadratic']
            error_forward=float(np.max(np.abs(actual-prediction[selected,arm])));forward_error=max(forward_error,error_forward)
            assert np.allclose(actual,prediction[selected,arm],atol=1e-5,rtol=1e-5)
            checks.append(dict(fold=fold,arm=name,forward_rows=selected.tolist(),maximum_absolute_error=error_forward))
            model.close();budget()
    for item in spec['source_pins']:check_ref(item)
    check_ref(done['predictions']);check_ref(done['audit_arrays'])
    report=dict(status='D32_INDEPENDENT_CPU_STATISTICS_NORMALIZERS_AND_FIXED40_FORWARD_CHECKS_PASSED',source_sha256=sha(__file__),
          spec_sha256=sha(spec_path),completion_sha256=sha(completion),all13230x5_query_metrics_verified=True,all1000_bootstrap_statistics_verified=True,
          all_FIT_normalizers_recomputed_from_physical_R4=True,models_SHA_and_fixed40_forward_per_model_type_verified=True,
          independent_refit_or_full13230_forward_claimed=False,forward_checks=checks,maximum_metric_error=error,
          maximum_forward_absolute_error=forward_error,maximum_quadratic_normalizer_absolute_error=normalizer_error,
          practical_gate_independently_reproduced=bool(supports),torch_imported=False,GPU_initialized=False,
          seconds=time.monotonic()-START,peak_extra_CPU_bytes=rss()-BASE,new_model_fits=0,new_images=0)
    write(output/'completion.json',report);print(json.dumps({k:report[k] for k in ['status','seconds','peak_extra_CPU_bytes','maximum_metric_error','maximum_forward_absolute_error']}),flush=True)


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('Finite audit termination')))
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--completion',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:run(a.spec,a.completion,a.output)
    except BaseException as error:
        if a.output.exists():write(a.output/('failure_'+str(time.time_ns())+'.json'),dict(error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-START))
        raise
