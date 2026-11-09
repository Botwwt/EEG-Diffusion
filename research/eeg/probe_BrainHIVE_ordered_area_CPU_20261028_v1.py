"""D38 complete-T, concept-held test of order beyond existing EEG information.

No D/test/image reads, ordinary loading, neural updates, or GPU context.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
import brainhive_ordered_area_20261028_v1 as M

START=time.monotonic()
ROOT=Path(__file__).resolve().parent

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def read(p):return json.loads(Path(p).read_bytes())

def checked(item):
    p=Path(item['path']);assert p.is_file() and sha(p)==item['sha256'],str(p)
    if 'bytes' in item:assert p.stat().st_size==item['bytes']
    return p

def ref(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))

def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def rss():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024

def save(p,x):
    with Path(p).open('xb') as f:np.save(f,x,allow_pickle=False)

def native_and_contributions(raw,state):
    W=state['proj_in.weight'];b=state['proj_in.bias'];a=state['block.layers.0.ff.w1.weight'];v=state['block.layers.0.ff.w2.weight']
    gamma=state['block.layers.0.norm.weight'];beta=state['block.layers.0.norm.bias']
    c=np.empty((len(raw),1024),np.float64);h=np.empty((len(raw),5,1024),np.float64)
    w=W.reshape(1024,17,250)
    with torch.no_grad():
        for lo in range(0,len(raw),256):
            x=torch.from_numpy(np.asarray(raw[lo:lo+256],np.float32).copy());z=F.linear(x,W,b)
            z=z+F.linear(F.gelu(F.linear(z,a)),v)
            cc=F.layer_norm(z,(1024,),gamma,beta,1e-6)
            spatial=x.reshape(-1,17,250)
            hh=torch.stack([F.linear(spatial[:,:,low:high].reshape(len(x),-1),w[:,:,low:high].reshape(1024,-1))
                    for low,high in zip(M.EDGES[:-1],M.EDGES[1:])],dim=1)
            assert (hh.sum(1)+b-F.linear(x,W,b)).abs().max()<8e-5
            c[lo:lo+len(x)]=cc.numpy();h[lo:lo+len(x)]=hh.numpy()
    return c,h

def run(spec_path):
    spec=read(spec_path);assert spec['status']=='D38_COMPLETE_T_ORDERED_AREA_IDENTIFIABILITY_LOCKED'
    assert spec['rank']==32 and spec['ridge_penalty_per_fit_row']==.01 and spec['seed']==1729
    for item in spec['source_pins']:checked(item)
    output=Path(spec['output']);assert not output.exists();output.mkdir(parents=True)
    base=rss();stage='INPUTS';records=[];torch.set_num_threads(2)
    def budget(label=None):
        nonlocal stage
        if label:stage=label
        assert time.monotonic()-START<840,'Finite900s CPU limit'
        assert rss()-base<10<<30,'Original10GiB extraCPU cap'
        assert not torch.cuda.is_initialized()
        assert sum(p.stat().st_size for p in output.rglob('*') if p.is_file())<2<<30
    try:
        upstream=read(checked(spec['D30_spec']));D33=read(checked(spec['D33_spec']))
        physical=read(checked(upstream['physical_R4_input_contract']))
        inputs={k:physical['files'][k] for k in ['trials.npy','targets.npy','index.jsonl']}
        for item in inputs.values():checked(item)
        assert inputs['trials.npy']['sha256']==D33['T_trials']['sha256']
        assert inputs['index.jsonl']['sha256']==D33['T_index']['sha256']
        index=[json.loads(line) for line in checked(inputs['index.jsonl']).read_text().splitlines()]
        assert len(index)==13230 and all(v['row']==i and v['partition']=='T' and v['subject_id']=='sub-01' for i,v in enumerate(index))
        from probe_BrainHIVE_leaveone_decision_GPU_20261028_v1 import validate_split
        _,source_rows,metadata,groups,folds=validate_split(checked(spec['split']),checked(spec['image_manifest']))
        for i,v in enumerate(index):
            assert v['array_row']==source_rows[i] and v['fold']==int(folds[i]) and v['concept_id']==metadata[source_rows[i]]['concept_id'] and v['stimulus_id']==metadata[source_rows[i]]['stimulus_id']
        trials=np.load(inputs['trials.npy']['path'],mmap_mode='r',allow_pickle=False)
        y=np.asarray(np.load(inputs['targets.npy']['path'],allow_pickle=False),np.float64)
        assert trials.shape==(13230,4,17,250) and trials.dtype==np.float32 and y.shape==(13230,1024)
        raw=np.empty((13230,4250),np.float64)
        for lo in range(0,len(raw),256):
            x=np.asarray(trials[lo:lo+256]);assert np.isfinite(x).all()
            # Match the native physical R4 FP32 averaging, then promote statistics.
            raw[lo:lo+len(x)]=x.mean(1,dtype=np.float32).reshape(len(x),4250)
        candidates=np.empty((13230,10),np.int64);labels=np.empty(13230,np.int8);wrong=np.arange(13230)
        for positions in groups.values():
            candidates[positions]=positions;labels[positions]=np.arange(10);wrong[positions]=np.roll(positions,-1)
        signs=(np.random.default_rng(1729).integers(0,2,size=13230)*2-1).astype(np.int8)
        predpath=output/'all13230x6_F_predictions.npy'
        predictions=np.lib.format.open_memmap(predpath,mode='w+',dtype=np.float32,shape=(13230,6,1024));predictions[:]=np.nan
        coverage=np.zeros((13230,6),np.uint8)
        for held_fold in (0,1):
            budget('ROLE_AND_NATIVE_FEATURES')
            held=np.flatnonzero(folds==held_fold);fit=np.flatnonzero(folds!=held_fold)
            assert not {index[i]['concept_id'] for i in held}&{index[i]['concept_id'] for i in fit}
            assert set(wrong[held])==set(held) and set(wrong[fit])==set(fit)
            role=upstream['roles']['oof'+str(held_fold)];assert role['held_fold']==held_fold
            checkpoint=checked(role['checkpoint']);state=torch.load(checkpoint,weights_only=True,map_location='cpu')['model']
            c,h=native_and_contributions(raw,state)
            budget('FIT_ONLY_PCA')
            projected,basis=M.fit_basis(h,fit,rank=32);del h
            area,scatter=M.area_and_scatter(projected)
            reverse,reverse_scatter=M.area_and_scatter(projected[:,::-1])
            assert np.allclose(reverse,-area,atol=1e-10) and np.allclose(reverse_scatter,scatter,atol=1e-10)
            assert area.shape==(13230,496) and np.isfinite(area).all()
            np.savez_compressed(output/f'fold{held_fold}_features.npz',native_conditions=c,projected_increments=projected,
              area=area,scatter=scatter,fit_rows=fit,held_rows=held,**basis)
            ymean=y[fit].mean(0);Y=torch.from_numpy(y[fit]-ymean);penalty=len(fit)*.01
            for arm in (0,1,2):
                budget('COMMON_BASELINE_FITS')
                common=c if arm==0 else np.concatenate((c,raw),axis=1)
                if arm==2:common=np.concatenate((common,scatter),axis=1)
                xf,xh,norm=M.standardize(common[fit],common[held]);del common
                X=torch.from_numpy(xf);L,coef=M.solve(torch,X,Y,penalty)
                p=(torch.from_numpy(xh)@coef).numpy()+ymean
                predictions[held,arm]=p.astype(np.float32);coverage[held,arm]+=1;predictions.flush()
                model=output/f'fold{held_fold}_{M.ARMS[arm]}_model.npz'
                np.savez_compressed(model,common_coefficient=coef.numpy(),input_mean=norm['mean'],input_scale=norm['scale'],target_mean=ymean,penalty=np.array(penalty))
                record=dict(held_fold=held_fold,arm=M.ARMS[arm],fit_rows=len(fit),held_rows=len(held),
                  common_dimensions=X.shape[1],extra_dimensions=0,model=ref(model),checkpoint=ref(checkpoint),PCA_variance_retained=basis['retained_variance'])
                records.append(record);write(output/f'fold{held_fold}_{M.ARMS[arm]}_fit.json',record)
                print(json.dumps(dict(stage='FIT_EXPORTED',held_fold=held_fold,arm=M.ARMS[arm],seconds=time.monotonic()-START)),flush=True)
                if arm!=2:del X,L,coef,xf,xh
            # All three added-area models share exactly the same c/raw/scatter baseline.
            for arm in (3,4,5):
                budget('CONDITIONAL_ORDER_FITS')
                z={3:area,4:area*signs[:,None],5:area[wrong]}[arm]
                zf,zh,znorm=M.standardize(z[fit],z[held]);Z=torch.from_numpy(zf)
                linear,extra=M.extend(torch,L,coef,X,Z,Y,penalty)
                p=(torch.from_numpy(xh)@linear+torch.from_numpy(zh)@extra).numpy()+ymean
                assert np.isfinite(p).all()
                predictions[held,arm]=p.astype(np.float32);coverage[held,arm]+=1;predictions.flush()
                model=output/f'fold{held_fold}_{M.ARMS[arm]}_model.npz'
                np.savez_compressed(model,common_coefficient=linear.numpy(),area_coefficient=extra.numpy(),
                  input_mean=norm['mean'],input_scale=norm['scale'],area_mean=znorm['mean'],area_scale=znorm['scale'],target_mean=ymean,penalty=np.array(penalty))
                record=dict(held_fold=held_fold,arm=M.ARMS[arm],fit_rows=len(fit),held_rows=len(held),common_dimensions=X.shape[1],
                   extra_dimensions=496,model=ref(model),checkpoint=ref(checkpoint),PCA_variance_retained=basis['retained_variance'])
                records.append(record);write(output/f'fold{held_fold}_{M.ARMS[arm]}_fit.json',record)
                print(json.dumps(dict(stage='FIT_EXPORTED',held_fold=held_fold,arm=M.ARMS[arm],seconds=time.monotonic()-START)),flush=True)
            del X,L,coef,xf,xh,Y,c,projected,area,scatter,reverse,reverse_scatter,linear,extra,Z,z,zf,zh
        assert np.all(coverage==1) and np.isfinite(predictions).all()
        budget('FULL_HELD_SCORING')
        values=np.stack([M.metrics(predictions[:,arm],y,candidates,labels) for arm in range(6)],axis=1)
        concept=np.stack([values[positions].mean(0) for positions in groups.values()])
        draws_index=np.random.default_rng(1729).integers(0,len(groups),size=(1000,len(groups)))
        all_summary,all_draws=M.summarize(concept,draws_index);comparisons={}
        for control in (1,2,4,5):
            difference=concept[:,3]-concept[:,control];difference[:,0]*=-1
            value,_=M.summarize(difference,draws_index)
            value['relative_MSE_improvement']=float((concept[:,control,0].mean()-concept[:,3,0].mean())/concept[:,control,0].mean())
            comparisons[M.ARMS[control]]=value
        conclusion=M.decision(comparisons)
        archive=output/'all_query_metrics_and_bootstraps.npz'
        np.savez_compressed(archive,metrics=values,concept_metrics=concept,bootstrap_means=all_draws,bootstrap_indices=draws_index,
            candidates=candidates,labels=labels,folds=folds,coverage=coverage,wrong_rows=wrong,random_signs=signs)
        budget('FINAL')
        for item in spec['source_pins']:checked(item)
        completion=dict(status='D38_COMPLETE13230_CONCEPT_HELD_ORDERED_AREA_DIAGNOSTIC',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            source=ref(Path(__file__).resolve()),spec=ref(spec_path),arms=M.ARMS,metric_order=M.METRICS,all_arms=all_summary,
            ordered_area_minus_controls=comparisons,decision=conclusion,analytical_models=12,model_records=records,
            predictions=ref(predpath),audit_arrays=ref(archive),complete13230_query_coverage_once=True,
            PCA_and_standardizers_and_decoders_FIT_only=True,native_role_excludes_query_concept=True,
            old_ordinary_used_or_changed=False,classic_signature_formula_claimed_novel=False,
            read_Dfit_Dcheck_V_or_official_values=False,new_neural_models=0,SGD_updates=0,new_GPU_jobs=0,new_images=0,
            maximum_output_bytes=2<<30,seconds=time.monotonic()-START,peak_extra_CPU_bytes=rss()-base,
            global_reversal_absorbable_by_refitting_not_used_as_control=True,
            same_concept_wrong_is_diagnostic_label_privileged_not_normal_inference=True)
        write(output/'completion.json',completion)
        print(json.dumps({k:completion[k] for k in ['status','decision','seconds','peak_extra_CPU_bytes']}),flush=True)
    except BaseException as error:
        write(output/f'failure_{time.time_ns()}.json',dict(status='D38_FAILURE_PRESERVED',error=repr(error),traceback=traceback.format_exc(),stage=stage,completed_models=len(records),seconds=time.monotonic()-START))
        raise

if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('Finite D38 CPU termination')))
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);run(p.parse_args().spec)
