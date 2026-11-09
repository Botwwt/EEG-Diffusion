"""D32: complete-T concept-held predictive test of cross-repeat quadratic EEG.

Five analytical decoders use identical raw mean4 first-order features; four
have the same128 bilinear features or controls. No native weights are updated,
ordinary is not read, and no Dfit/Dcheck/V/image or automatic fit admission.
"""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='2'
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import signal
import socket
import time
import traceback
import numpy as np
import brainhive_crossrepeat_quadratic_20261028_v1 as Q

ROOT=Path(__file__).resolve().parent
ARMS=('linear_mean4','mean4_naive_quadratic','cross_repeat_U','noise_cross_covariance','same_concept_wrong_U')
METRICS=('F_MSE','correct_F_cosine','same_concept_Top1','same_concept_2AFC','correct_minus_other9_cosine')
START=time.monotonic()


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_bytes())


def checked(item):
    path=Path(item['path']);assert path.is_file() and sha(path)==item['sha256'],str(path)
    if 'bytes' in item:assert path.stat().st_size==item['bytes']
    return path


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())


def rss():
    return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024


class Budget:
    def __init__(self,output):self.output=output;self.base=rss();self.stage='VERIFY_INPUTS';self.torch=None
    def check(self,stage=None):
        if stage:self.stage=stage
        assert time.monotonic()-START<840,'Finite900second CPU job reserve'
        assert rss()-self.base<10<<30,'Original10GiB extra CPU cap'
        if self.torch is not None:assert not self.torch.cuda.is_initialized(),'No GPU context allowed'
        if self.output.exists():assert sum(p.stat().st_size for p in self.output.rglob('*') if p.is_file())<1<<30


def interval(values,indices):
    mean=values.mean(0);draws=np.stack([values[indices[lo:lo+20]].mean(1) for lo in range(0,len(indices),20)]).reshape((len(indices),)+values.shape[1:])
    lower,upper=np.quantile(draws,[.025,.975],axis=0)
    return dict(mean=mean.tolist(),descriptive_95_lower=lower.tolist(),descriptive_95_upper=upper.tolist()),draws


def run(spec_path):
    spec=read(spec_path);assert spec['status']=='D32_CROSS_REPEAT_QUADRATIC_PREDICTIVE_DIAGNOSTIC_LOCKED'
    assert spec['seed']==1729 and spec['features']==128 and spec['ridge_penalty_per_fit_row']==.01
    output=Path(spec['output']);assert not output.exists();output.mkdir(parents=True)
    budget=Budget(output);completed=[]
    try:
        for item in spec['source_pins']:checked(item)
        upstream=read(checked(spec['upstream_D30_spec']))
        physical=read(checked(upstream['physical_R4_input_contract']))
        assert physical['status']=='PHYSICAL_R4_T_INPUTS_READY' and physical['export_shape']==[13230,4,17,250]
        refs=physical['files']
        for name in ('trials.npy','targets.npy','index.jsonl'):checked(refs[name])
        assert refs['trials.npy']['sha256']=='b804a105822bfab5eeb853b6942d25109eaf88f3f8a1eb5b503b444a4dc0d393'
        assert refs['targets.npy']['sha256']=='363e8e74514ca8d1bcbfb29afc2023ed4a5c3a6c4f49bf8b1134e7ae7a090336'
        assert refs['index.jsonl']['sha256']=='a12c2758f8555223e08022e5b4a195da37ad3dac37cb6e2806aaf17bac014e67'
        from probe_BrainHIVE_leaveone_decision_GPU_20261028_v1 import validate_split
        split,rows,metadata,groups,folds=validate_split(checked(spec['split']),checked(spec['image_manifest']))
        index=[json.loads(line) for line in Path(refs['index.jsonl']['path']).read_text().splitlines()]
        for i,row in enumerate(index):
            assert row['row']==i and row['array_row']==rows[i] and row['partition']=='T' and row['subject_id']=='sub-01'
            assert row['concept_id']==metadata[rows[i]]['concept_id'] and row['stimulus_id']==metadata[rows[i]]['stimulus_id'] and row['fold']==int(folds[i])
        assert len(index)==13230
        authorization=read(checked(spec['authorization']))
        assert authorization['formal_old_D_training_resumed'] is False and authorization['existing_ordinary_baseline_bytes_immutable']
        import torch
        torch.set_num_threads(2);budget.torch=torch
        assert not torch.cuda.is_initialized()
        trials=np.load(refs['trials.npy']['path'],mmap_mode='r',allow_pickle=False)
        targets=np.load(refs['targets.npy']['path'],mmap_mode='r',allow_pickle=False)
        assert trials.shape==(13230,4,17,250) and trials.dtype==np.float32
        assert targets.shape==(13230,1024) and targets.dtype==np.float32 and np.isfinite(targets).all()
        raw_mean=np.empty((13230,4250),np.float64)
        mean32_relative=0.
        for lo in range(0,13230,256):
            x=np.asarray(trials[lo:lo+256]);assert np.isfinite(x).all()
            m=x.astype(np.float64).mean(1).reshape(len(x),4250);raw_mean[lo:lo+len(x)]=m
            m32=x.mean(1,dtype=np.float32).reshape(len(x),4250)
            mean32_relative=max(mean32_relative,float((np.linalg.norm(m-m32,axis=1)/np.maximum(np.linalg.norm(m,axis=1),1e-30)).max()))
            budget.check('PHYSICAL_MEAN4')
        generator=np.random.default_rng(1729)
        A=(generator.integers(0,2,size=(4250,128),dtype=np.int8)*2-1).astype(np.float64)/np.sqrt(4250.)
        B=(generator.integers(0,2,size=(4250,128),dtype=np.int8)*2-1).astype(np.float64)/np.sqrt(4250.)
        B[:,:64]=A[:,:64]
        candidates=np.empty((13230,10),np.int64);labels=np.empty(13230,np.int8);wrong=np.arange(13230)
        for positions in groups.values():
            candidates[positions]=positions;labels[positions]=np.arange(10);wrong[positions]=np.roll(positions,-1)
        predictions_path=output/'all13230x5_F_predictions.npy'
        predictions=np.lib.format.open_memmap(predictions_path,mode='w+',dtype=np.float32,shape=(13230,5,1024))
        predictions[:]=np.nan;coverage=np.zeros((13230,5),np.uint8)
        write(output/'provenance.json',dict(spec_sha256=sha(spec_path),source_sha256=sha(__file__),host=socket.gethostname(),
              exact_input_refs=refs,arms=ARMS,primary='cross_repeat_U minus same-capacity mean4_naive_quadratic',
              fixed_projection_distribution='Rademacher/sqrt4250;64 diagonal+64 independent bilinear pairs; generated once seed1729',
              first_order_features=4250,extra_features=128,physical_repeats=4,repeat_independence_or_neural_origin_established=False,
              FIT_only_standardizers=True,outer_concepts_excluded_from_all_new_fit=True,ordinary_used=False,
              method_is_analytical_input_identifiability_probe_not_deployed_BrainMLP_change=True))
        for held_fold in (0,1):
            held=np.flatnonzero(folds==held_fold);fit=np.flatnonzero(folds!=held_fold)
            assert not set(index[i]['concept_id'] for i in fit)&set(index[i]['concept_id'] for i in held)
            assert set(wrong[fit])==set(fit) and set(wrong[held])==set(held)
            xf,xh,normalizer=Q.fit_standardize(raw_mean[fit],raw_mean[held])
            mu,scale=normalizer['mean'],normalizer['scale']
            naive=np.empty((13230,128),np.float64);cross=np.empty_like(naive);noise=np.empty_like(naive);identity_error=0.;negative=0
            for lo in range(0,13230,128):
                raw=np.asarray(trials[lo:lo+128]).astype(np.float64).reshape(-1,4,4250)
                standardized=(raw-mu)/scale
                za=(standardized.reshape(-1,4250)@A).reshape(len(raw),4,128)
                zb=(standardized.reshape(-1,4250)@B).reshape(len(raw),4,128)
                n,u,s,error=Q.features(za,zb);count=len(raw)
                naive[lo:lo+count]=n;cross[lo:lo+count]=u;noise[lo:lo+count]=s
                identity_error=max(identity_error,error);negative+=int((u[:,:64]<0).sum())
                if lo==0:
                    direct=sum(za[:,r]*zb[:,s] for r in range(4) for s in range(4) if r!=s)/12
                    assert np.allclose(direct,u,atol=1e-10,rtol=1e-11)
                budget.check('FIXED_BILINEAR_FEATURES')
            ymean=np.asarray(targets[fit],np.float64).mean(0);yf=np.asarray(targets[fit],np.float64)-ymean
            X=torch.from_numpy(xf);Y=torch.from_numpy(yf);penalty=len(fit)*.01
            G=X.T@X;G.diagonal().add_(penalty)
            L=torch.linalg.cholesky(G);base=torch.cholesky_solve(X.T@Y,L);del G
            for arm,name in enumerate(ARMS):
                budget.check('ANALYTICAL_JOINT_FIT')
                if arm==0:
                    coefficient=base;qcoefficient=None;qnorm=None
                    predicted=(torch.from_numpy(xh)@coefficient).numpy()+ymean
                else:
                    extra={1:naive,2:cross,3:noise,4:cross[wrong]}[arm]
                    zf,zh,qnorm=Q.fit_standardize(extra[fit],extra[held])
                    Z=torch.from_numpy(zf)
                    coefficient,qcoefficient=Q.joint_ridge_extension(torch,L,base,X,Z,Y,penalty)
                    predicted=(torch.from_numpy(xh)@coefficient+torch.from_numpy(zh)@qcoefficient).numpy()+ymean
                assert np.isfinite(predicted).all()
                predictions[held,arm]=predicted.astype(np.float32);coverage[held,arm]+=1;predictions.flush()
                model_path=output/('fold'+str(held_fold)+'_'+name+'_analytical_model.npz')
                arrays=dict(linear=coefficient.numpy(),input_mean=mu,input_scale=scale,target_mean=ymean,A=A,B=B,penalty=np.array(penalty))
                if qcoefficient is not None:arrays.update(quadratic=qcoefficient.numpy(),quadratic_mean=qnorm['mean'],quadratic_scale=qnorm['scale'])
                np.savez_compressed(model_path,**arrays)
                record=dict(held_fold=held_fold,arm=name,fit_rows=len(fit),held_rows=len(held),penalty=penalty,
                    model=dict(path=str(model_path),sha256=sha(model_path),bytes=model_path.stat().st_size),
                    maximum_feature_identity_relative_error=identity_error,negative_diagonal_U_entries=negative,
                    fit_concepts=len({index[i]['concept_id'] for i in fit}),held_concepts=len({index[i]['concept_id'] for i in held}))
                completed.append(record);write(output/('fold'+str(held_fold)+'_'+name+'_fit.json'),record)
                print(json.dumps(dict(stage='FITTED_AND_EXPORTED_HELD',held_fold=held_fold,arm=name,seconds=time.monotonic()-START)),flush=True)
            del X,Y,L,base,xf,xh,yf,naive,cross,noise
        assert np.all(coverage==1) and np.isfinite(predictions).all()
        budget.check('FULL_OOF_SCORING')
        metrics=np.empty((13230,5,5),np.float64)
        y=np.asarray(targets,np.float64)
        yn=y/np.maximum(np.linalg.norm(y,axis=1,keepdims=True),1e-30)
        for lo in range(0,13230,128):
            for arm in range(5):
                # Candidate targets come from the same fixed concept, including
                # targets outside this score chunk; they never enter inference.
                p=np.asarray(predictions[lo:lo+128,arm],np.float64)
                pn=p/np.maximum(np.linalg.norm(p,axis=1,keepdims=True),1e-30)
                similarities=np.einsum('nd,nkd->nk',pn,yn[candidates[lo:lo+len(p)]])
                selected=similarities[np.arange(len(p)),labels[lo:lo+len(p)]]
                top=(selected==similarities.max(1))/(similarities==similarities.max(1)[:,None]).sum(1)
                afc=((selected[:,None]>similarities).sum(1)+.5*(selected[:,None]==similarities).sum(1)-.5)/9
                metrics[lo:lo+len(p),arm]=np.stack((np.square(p-y[lo:lo+len(p)]).mean(1),selected,top,afc,selected-(similarities.sum(1)-selected)/9),axis=-1)
            budget.check()
        concept=np.stack([metrics[positions].mean(0) for positions in groups.values()])
        draws_index=np.random.default_rng(1729).integers(0,1323,size=(1000,1323))
        all_results,draws=interval(concept,draws_index);comparisons={}
        for control in (0,1,3,4):
            difference=concept[:,2]-concept[:,control];difference[:,0]*=-1
            item,_=interval(difference,draws_index)
            item['relative_MSE_improvement']=float((concept[:,control,0].mean()-concept[:,2,0].mean())/concept[:,control,0].mean())
            comparisons[ARMS[control]]=item
        base_c= comparisons['linear_mean4'];naive_c=comparisons['mean4_naive_quadratic']
        supports=(base_c['relative_MSE_improvement']>=.005 and naive_c['relative_MSE_improvement']>=.005
            and base_c['mean'][2]>=.01 and naive_c['mean'][2]>=0
            and all(x['descriptive_95_lower'][0]>0 for x in comparisons.values()))
        archive=output/'all_query_metrics_and_bootstraps.npz'
        np.savez_compressed(archive,metrics=metrics,concept_metrics=concept,bootstrap_means=draws,bootstrap_indices=draws_index,
                            candidate_rows=candidates,labels=labels,folds=folds,coverage=coverage,wrong_U_rows=wrong)
        for item in spec['source_pins']:checked(item)
        for name in ('trials.npy','targets.npy','index.jsonl'):checked(refs[name])
        budget.check('FINAL')
        report=dict(status='D32_COMPLETE_FULL_T_CROSS_REPEAT_QUADRATIC_PREDICTIVE_DIAGNOSTIC',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             source_sha256=sha(__file__),spec_sha256=sha(spec_path),arms=ARMS,metric_order=METRICS,all_arms=all_results,U_minus_controls=comparisons,
             analytical_models=len(completed),model_records=completed,predictions=dict(path=str(predictions_path),sha256=sha(predictions_path),bytes=predictions_path.stat().st_size),
             audit_arrays=dict(path=str(archive),sha256=sha(archive),bytes=archive.stat().st_size),all13230_query_coverage_exactly_one=True,
             physical_mean64_vs_old_mean32_max_relative_error=mean32_relative,models_fit_exclude_current_concept=True,
             practical_predictive_gate_passed=bool(supports),decision='PREDICTIVE_SUPPORT_ONLY_REQUIRES_SEPARATE_ARCHITECTURE_AND_IMAGE_TEST' if supports else 'STOP_THIS_FIXED_RANDOM_BILINEAR_INSTANCE_NO_AUTOMATIC_TRAINING_OR_IMAGES',
             unbiasedness_is_conditional_on_zero_mean_uncorrelated_repeat_errors=True,these_assumptions_empirically_proven=False,
             ordinary_used_or_changed=False,new_SGD_updates=0,new_GPU_jobs=0,new_images=0,classic_formula_claimed_novel=False,
             Dfit_Dcheck_V_or_official_effects_read=False,automatic_following_fit_or_images=False,
             seconds=time.monotonic()-START,peak_extra_CPU_bytes=rss()-budget.base,torch_CUDA_initialized=torch.cuda.is_initialized())
        write(output/'completion.json',report)
        print(json.dumps({k:report[k] for k in ['status','analytical_models','practical_predictive_gate_passed','decision','seconds','peak_extra_CPU_bytes']}),flush=True)
    except BaseException as error:
        write(output/('failure_'+str(time.time_ns())+'.json'),dict(status='D32_CPU_DIAGNOSTIC_FAILED_PRESERVED',error=repr(error),traceback=traceback.format_exc(),stage=budget.stage,completed_models=completed,seconds=time.monotonic()-START,new_GPU_jobs=0,new_images=0))
        raise


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('Finite CPU diagnostic termination')))
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);a=p.parse_args();run(a.spec)
