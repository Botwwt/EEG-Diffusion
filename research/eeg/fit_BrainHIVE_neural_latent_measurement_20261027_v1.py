"""D06: an independently checked, affine image-latent to neural measurement.

Only outer-FIT scaled latent and group-A EEG fit the new model.  B-held EEG
serves exclusively as an independent neural evaluation.  A failed fixed gate
does not export a generation operator and must not launch image generation.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import signal
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import fit_BrainHIVE_raw_neural_recoverability_20261027_v1 as N

ROOT=Path(__file__).resolve().parent
SCHEMA='BrainHIVE_neural_latent_measurement_20261027_v1'
PASSED='COMPLETE_CROSSBRAIN_NEURAL_LATENT_MEASUREMENT_GATE_PASSED'
FAILED='STOP_CROSSBRAIN_NEURAL_LATENT_FORWARD_GATE'
FLATTEN='C-order channel0..3,y0..63,x0..63; native VAE MODE * one scaling_factor'
LATENT_SHA='eda12be7eeba9308437f9d58e5809f198dcc0a730fea8d63b2cd7158f88d0039'
RECEIPT_SHA='a6fb379d25fe3738c47c036fa591c0603f952ba9e119763c487151462a4dae6f'
sha,ref,atomic=N.sha,N.ref,N.atomic


def verified(binding):
    path=Path(binding['path']);assert sha(path)==binding['sha256'],f'SHA mismatch {path}'
    return path


def on_device(value,device):
    if torch.is_tensor(value):return value.to(device)
    if isinstance(value,dict):return {k:on_device(v,device) for k,v in value.items()}
    if isinstance(value,list):return [on_device(v,device) for v in value]
    return value


def fit_latent_PCA(values,dimensions=128):
    """No coordinate standardization: centered scaled-native latent PCA."""
    mean=values.mean(0);centered=values-mean
    torch.manual_seed(1729)
    _,singular,basis=torch.pca_lowrank(centered,q=min(dimensions+16,*centered.shape),center=False,niter=2)
    basis=basis[:,:dimensions]
    assert basis.shape[1]==dimensions
    # Evaluate the frozen PCA in double precision so exported affine composition
    # represents the exact same mathematical predictor, rather than a new model.
    pc=(values.double()-mean.double())@basis.double()
    pc_mean=pc.mean(0);pc_std=pc.std(0,unbiased=True).clamp_min(1e-4)
    return dict(latent_mean=mean,PCA_basis=basis,PC_mean=pc_mean,PC_std=pc_std,
        explained_variance=float(singular[:dimensions].square().sum()/centered.square().sum().clamp_min(1e-12)))


def transform(values,pca):
    return (((values.double()-pca['latent_mean'].double())@pca['PCA_basis'].double())-pca['PC_mean'])/pca['PC_std']


def compose_affine(pca,ridge):
    weights=ridge['weights'].double()
    H=(pca['PCA_basis'].double()/pca['PC_std'][None])@weights
    b=ridge['target_center']-(pca['PC_mean']/pca['PC_std']+ridge['feature_center'])@weights-pca['latent_mean'].double()@H
    return H,b


def regularized_inverse(H):
    gram=H.T@H;trace=torch.trace(gram)
    assert torch.isfinite(trace) and trace>1e-20,'STOP: degenerate learned forward operator'
    penalty=.1*trace/H.shape[1]
    system=gram+penalty*torch.eye(H.shape[1],dtype=H.dtype,device=H.device)
    K=torch.cholesky_solve(H.T,torch.linalg.cholesky(system))
    assert torch.isfinite(K).all()
    return K,float(penalty)


def fold_gate(metrics):
    return bool(metrics['paired_cosine_gap']>.01 and metrics['sameconcept10_top1']>=.12)


def pooled_gate(interval):
    return bool(interval['paired_cosine_gap95'][0]>0 and interval['paired_top1_gap95'][0]>0)


def save_npy(path,value):
    path=Path(path)
    with path.open('xb') as stream:
        np.save(stream,np.asarray(value),allow_pickle=False);stream.flush();os.fsync(stream.fileno())


def self_test():
    torch.set_num_threads(2);torch.manual_seed(1729)
    x=torch.randn(100,24,dtype=torch.float64)
    basis,_=torch.linalg.qr(torch.randn(24,8,dtype=torch.float64))
    pca=dict(latent_mean=torch.randn(24,dtype=torch.float64),PCA_basis=basis,
        PC_mean=torch.randn(8,dtype=torch.float64),PC_std=torch.rand(8,dtype=torch.float64)+.5)
    y=torch.randn(100,3,dtype=torch.float64)
    ridge=N.ridge_fit(transform(x,pca),y)
    H,b=compose_affine(pca,ridge)
    explicit=N.ridge_predict(transform(x,pca),ridge)
    assert torch.allclose(x@H+b,explicit,atol=1e-12,rtol=1e-12)
    K,penalty=regularized_inverse(H)
    assert abs(penalty-.1*float(torch.trace(H.T@H))/3)<1e-15
    assert torch.allclose((H.T@H+penalty*torch.eye(3,dtype=H.dtype))@K,H.T,atol=1e-12)
    wanted=torch.randn(100,3,dtype=torch.float64)
    corrected=x+.25*(wanted-(x@H+b))@K
    assert ((corrected@H+b)-wanted).norm()<(explicit-wanted).norm()
    try:
        regularized_inverse(torch.zeros_like(H));raise RuntimeError('zero forward operator accepted')
    except AssertionError:pass
    assert not fold_gate(dict(paired_cosine_gap=.01,sameconcept10_top1=.12))
    assert fold_gate(dict(paired_cosine_gap=.0101,sameconcept10_top1=.12))
    assert not fold_gate(dict(paired_cosine_gap=.2,sameconcept10_top1=.119))
    assert not pooled_gate(dict(paired_cosine_gap95=[0,.1],paired_top1_gap95=[.01,.1]))
    assert pooled_gate(dict(paired_cosine_gap95=[.01,.1],paired_top1_gap95=[.01,.1]))
    fitted_input=torch.randn(90,32)
    fitted=fit_latent_PCA(fitted_input,8)
    assert transform(torch.randn(7,32),fitted).shape==(7,8)
    train_scores=transform(fitted_input,fitted)
    assert torch.allclose(train_scores.mean(0),torch.zeros(8,dtype=torch.float64),atol=1e-12)
    assert torch.allclose(train_scores.std(0,unbiased=True),torch.ones(8,dtype=torch.float64),atol=1e-12)
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(status='CPU_NEURAL_LATENT_MATH_AND_GATE_CHECKS_PASSED',tests=10,CUDA_initialized=False,
        exact_double_affine_composition_max_abs=float((x@H+b-explicit).abs().max()),
        scope='synthetic affine/ridge/inverse/gate checks only; no actual latent or EEG fit')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','d04-models-binding','latent-receipt','latents','spec','output'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--gpu-job-name');parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--max-seconds',type=float,default=600);parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test();return 0
    assert all((args.inputs,args.d04_models_binding,args.latent_receipt,args.spec,args.output,args.gpu_job_name))
    assert args.device=='cuda:0' and 0<args.max_seconds<=600 and not args.output.exists()
    args.output.mkdir(parents=True);invoked=time.monotonic();stopped=[False]
    signal.signal(signal.SIGTERM,lambda number,frame:stopped.__setitem__(0,True))
    import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
    D.torch=torch
    document=dict(gpu_job_name=args.gpu_job_name,maximum_seconds=args.max_seconds,
        maximum_extra_CPU_bytes=10<<30,maximum_GPU_bytes=20<<30,minimum_CPU_headroom_bytes=10<<30,
        maximum_output_bytes=128<<20,resource_authorization=ref(ROOT/'configs/resource_authorization_20261020_v3.json'))
    wrapper=D.verify_wrapper(document);budget=D.Budget(document)
    def check():
        assert not stopped[0] and time.monotonic()-invoked<args.max_seconds,'Finite D06 fit limit'
        budget.check()
    def log(value):
        value=value|dict(seconds=time.monotonic()-invoked)
        with (args.output/'progress.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(value,allow_nan=False)+'\n');stream.flush()
        print(json.dumps(value),flush=True)
    try:
        torch.set_num_threads(2);torch.manual_seed(1729);torch.cuda.set_device(torch.device(args.device))
        torch.cuda.reset_peak_memory_stats();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        spec=json.loads(args.spec.read_bytes());assert spec['schema']==SCHEMA and spec['fit']['ridge_lambda']==.1
        assert sha(N.__file__)==spec['D04_source_sha256']
        inputs=json.loads(args.inputs.read_bytes());assert inputs['status']=='COMPLETE_RAW_NEURAL_T17_ALL10_INPUTS'
        assert inputs['rows']==13230 and inputs['concepts']==1323
        index=Path(inputs['prediction_index']);assert sha(index)==inputs['prediction_index_sha256']==spec['prediction_index_sha256']
        items=N.read_index(index);assert len(items)==13230 and [row['row'] for row in items]==list(range(13230))
        grouped={}
        for row in items:grouped.setdefault(row['concept_id'],[]).append(row)
        assert len(grouped)==1323 and all(len(rows)==10 and len({row['fold'] for row in rows})==1 for rows in grouped.values())
        binding=json.loads(args.d04_models_binding.read_bytes())
        assert binding['status']=='D04_COMPLETE_FIXED_GATE_AND_MODELS_BOUND' and binding['fixed_practical_screen_gate_passed'] is True
        d04_completion_path=verified(binding['completion']);d04_summary_path=verified(binding['summary'])
        completion=json.loads(d04_completion_path.read_bytes());d04=json.loads(d04_summary_path.read_bytes())
        assert completion['status']==d04['status']=='COMPLETE_RAW_NEURAL_RECOVERABILITY_SCREEN'
        assert completion['summary']['sha256']==sha(d04_summary_path) and d04['fixed_practical_screen_gate_passed'] is True
        assert d04['group_A']==list(N.GROUP_A) and d04['group_B']==list(N.GROUP_B)
        query_path=verified(d04['query_predictions']);query=np.load(query_path,mmap_mode='r',allow_pickle=False)
        assert query.shape==(13230,16) and query.dtype==np.float32 and np.isfinite(query).all()
        assert sha(args.latent_receipt)==RECEIPT_SHA==spec['target']['latent_receipt_sha256']
        latent_receipt=json.loads(args.latent_receipt.read_bytes())
        assert latent_receipt['status']=='COMPLETE_ALL13230_ALLOWED_T_NATIVE_VAE_MODE_LATENTS'
        assert latent_receipt['shape']==[13230,4,64,64] and latent_receipt['dtype']=='float32'
        assert latent_receipt['sha256']==LATENT_SHA==spec['target']['latent_bank_sha256']
        assert latent_receipt['registration']['VAE_scaling_factor']==.13025
        latent_path=args.latents or Path(latent_receipt['path']);assert sha(latent_path)==LATENT_SHA
        latent_bank=np.load(latent_path,mmap_mode='r',allow_pickle=False)
        assert latent_bank.shape==(13230,4,64,64) and latent_bank.dtype==np.float32
        source_rows=[int(row['source_row']) for row in items]
        assert set(source_rows)==set(latent_receipt['array_rows']) and len(set(source_rows))==13230
        lookup={source:i for i,source in enumerate(latent_receipt['array_rows'])}
        latent_indices=[lookup[source] for source in source_rows]
        raw_bindings={name:{k:inputs['subjects'][name][k] for k in ('path','sha256')} for name in N.GROUP_A+N.GROUP_B}
        for source in raw_bindings.values():check();verified(source)
        controls={key:ref(path) for key,path in [('inputs',args.inputs),('D04_binding',args.d04_models_binding),
            ('D04_completion',d04_completion_path),('D04_summary',d04_summary_path),('query_measurements',query_path),
            ('index',index),('latent_receipt',args.latent_receipt),('latents',latent_path),('spec',args.spec),('source',Path(__file__)),('D04_source',Path(N.__file__))]}
        atomic(args.output/'provenance.json',dict(schema=SCHEMA,control_bindings=controls,raw_source_bindings=raw_bindings,wrapper=wrapper,
            new_fit_uses_outer_fit_A_and_latent_only=True,B_held_evaluation_only=True,B_fit_only_used_previously_in_D04_state=True,
            additional_subject_EEG_at_inference=False,no_query_target_access=True,outer_held_used_for_selection=False,
            outer_held_used_for_hyperparameter_selection=False,outer_held_used_for_fixed_generation_gate=True,
            latent_scaling_factor=.13025,latent_scale_already_applied_once=True,no_visual_F_loaded=True,
            learned_surrogate_off_manifold_validity_established=False,independent_noise_or_exact_Bayes_claim=False))
        tables=None;coverage=np.zeros(13230,np.int32);fold_results=[];states={};operator_data={}
        prediction_comparison={key:np.empty(13230,np.float64) for key in ('queryAhat_norm','image_predictedA_norm',
            'queryAhat_minus_image_predictedA_RMS','queryAhat_image_prediction_cosine')}
        for fold in (0,1):
            check();checkpoint_path=verified(binding['fold_models'][str(fold)])
            checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
            fit_positions=[i for i,row in enumerate(items) if row['fold']!=fold]
            held_positions=[i for i,row in enumerate(items) if row['fold']==fold]
            assert checkpoint['outer_fold']==fold
            assert checkpoint['fit_source_rows']==[source_rows[i] for i in fit_positions]
            assert checkpoint['held_source_rows']==[source_rows[i] for i in held_positions]
            # Only A states fit the forward model. B states/rotation are retained
            # exclusively to compute the held independent evaluation afterwards.
            A_state=on_device(dict(subject_models={name:checkpoint['subject_models'][name] for name in N.GROUP_A},MCCA=checkpoint['MCCA_A']),args.device)
            B_state=on_device(dict(subject_models={name:checkpoint['subject_models'][name] for name in N.GROUP_B},MCCA=checkpoint['MCCA_B'],rotation=checkpoint['A_to_B_rotation']),args.device)
            del checkpoint
            views=[]
            for name in N.GROUP_A:
                check();array=np.load(raw_bindings[name]['path'],mmap_mode='r',allow_pickle=False)
                assert array.shape==(13230,17,250) and array.dtype==np.float32
                raw=torch.from_numpy(np.array(array[fit_positions],copy=True).reshape(len(fit_positions),4250)).to(args.device);del array
                views.append(N.subject_transform(raw,A_state['subject_models'][name]).double());del raw
            A_fit,_=N.group_transform(views,A_state['MCCA']);del views,A_state
            latent_fit=torch.from_numpy(np.array(latent_bank[[latent_indices[i] for i in fit_positions]],copy=True).reshape(len(fit_positions),16384)).to(args.device)
            assert torch.isfinite(latent_fit).all()
            pca=fit_latent_PCA(latent_fit);features=transform(latent_fit,pca);del latent_fit
            ridge=N.ridge_fit(features,A_fit);del features,A_fit
            H,b=compose_affine(pca,ridge)
            # Check an exact double-precision composition before evaluating held
            # targets. These inputs contain no held brain measurement.
            probe=torch.from_numpy(np.array(latent_bank[[latent_indices[i] for i in fit_positions[:8]]],copy=True).reshape(8,16384)).to(args.device)
            explicit=N.ridge_predict(transform(probe,pca),ridge);affine=probe.double()@H+b
            composition_error=float((explicit-affine).abs().max())
            assert torch.allclose(explicit,affine,atol=1e-8,rtol=1e-8),'Affine export changed the fitted pipeline'
            del probe,explicit,affine
            log(dict(stage='OUTER_FIT_LATENT_FORWARD_READY',outer_fold=fold,train_rows=len(fit_positions),
                PCA_explained_variance=pca['explained_variance'],affine_composition_max_abs=composition_error))
            latent_held=torch.from_numpy(np.array(latent_bank[[latent_indices[i] for i in held_positions]],copy=True).reshape(len(held_positions),16384)).to(args.device)
            predicted_A=latent_held.double()@H+b;del latent_held
            views=[]
            for name in N.GROUP_B:
                check();array=np.load(raw_bindings[name]['path'],mmap_mode='r',allow_pickle=False)
                assert array.shape==(13230,17,250) and array.dtype==np.float32
                raw=torch.from_numpy(np.array(array[held_positions],copy=True).reshape(len(held_positions),4250)).to(args.device);del array
                views.append(N.subject_transform(raw,B_state['subject_models'][name]).double());del raw
            B_held,_=N.group_transform(views,B_state['MCCA']);del views
            # Both are predictions in one frozen A coordinate system. Their
            # mismatch is reported, not interpreted as measurement uncertainty.
            query_held=torch.from_numpy(np.array(query[held_positions],copy=True)).to(args.device).double()
            comparisons=dict(queryAhat_norm=query_held.norm(dim=1),image_predictedA_norm=predicted_A.norm(dim=1),
                queryAhat_minus_image_predictedA_RMS=(query_held-predicted_A).square().mean(1).sqrt(),
                queryAhat_image_prediction_cosine=torch.nn.functional.cosine_similarity(query_held,predicted_A,dim=1))
            for key,value in comparisons.items():prediction_comparison[key][held_positions]=value.cpu().numpy()
            scores=N.paired_metrics(predicted_A@B_state['rotation'],B_held,[items[i] for i in held_positions])
            if tables is None:tables={key:np.empty(13230,np.float64) for key in scores}
            for key,value in scores.items():tables[key][held_positions]=value
            coverage[held_positions]+=1
            metrics={key:float(value.mean()) for key,value in scores.items()}
            result=dict(outer_fold=fold,fit_rows=len(fit_positions),held_rows=len(held_positions),metrics=metrics,
                practical_fold_gate_passed=fold_gate(metrics),PCA_explained_variance=pca['explained_variance'],
                affine_composition_max_abs=composition_error,all16_held_correlations=N.correlations(predicted_A@B_state['rotation'],B_held),
                queryAhat_vs_image_predictedA_all16_correlations=N.correlations(query_held,predicted_A),
                prediction_comparison_means={key:float(value.mean()) for key,value in comparisons.items()},
                predictedA_RMS=float(predicted_A.square().mean().sqrt()),D04_model=binding['fold_models'][str(fold)])
            fold_results.append(result);atomic(args.output/f'fold{fold}_summary.json',result)
            states[str(fold)]=N.cpu_value(dict(PCA=pca,ridge=ridge,outer_fold=fold,
                fit_source_rows=[source_rows[i] for i in fit_positions],held_source_rows=[source_rows[i] for i in held_positions]))
            operator_data[str(fold)]=(H.cpu(),b.cpu())
            log(dict(stage='OUTER_HELD_INDEPENDENT_NEURAL_FORWARD_GATE',**result))
            del pca,ridge,H,b,B_state,B_held,predicted_A,query_held,comparisons
            torch.cuda.empty_cache()
        assert np.all(coverage==1) and tables is not None
        interval=N.bootstrap(items,tables);passed=all(row['practical_fold_gate_passed'] for row in fold_results) and pooled_gate(interval)
        per_query=args.output/'per_query.jsonl.gz'
        with gzip.open(per_query,'wt',encoding='utf-8') as stream:
            for i,row in enumerate(items):
                stream.write(json.dumps(row|{key:float(value[i]) for key,value in tables.items()}|
                    {'queryAhat_vs_image_prediction':{key:float(value[i]) for key,value in prediction_comparison.items()}},allow_nan=False)+'\n')
        exported={}
        if passed:
            for fold,(H,b) in operator_data.items():
                K,penalty=regularized_inverse(H)
                directory=args.output/f'fold{fold}';directory.mkdir()
                singular=torch.linalg.svdvals(H)
                assert torch.isfinite(singular).all() and singular.min()>=0 and singular.max()>0
                entry=dict(ridge_lambda=penalty,H_singular_values=singular.tolist(),H_gram_trace=float(torch.trace(H.T@H)),
                    regularized_gram_condition_number=float((singular.max().square()+penalty)/(singular.min().square()+penalty)))
                for name,value in [('H',H),('b',b),('K',K)]:
                    path=directory/f'{name}.npy';save_npy(path,value.numpy())
                    entry.update({f'{name}_path':str(path.resolve()),f'{name}_sha256':sha(path),f'{name}_shape':list(value.shape)})
                exported[fold]=entry
                torch.save(states[fold],directory/'forward_fit.pt')
        for source in controls.values():check();verified(source)
        for source in raw_bindings.values():check();verified(source)
        for source in binding['fold_models'].values():verified(source)
        assert sum(path.stat().st_size for path in args.output.rglob('*') if path.is_file())<128<<20
        summary=dict(status=PASSED if passed else FAILED,schema=SCHEMA,rows=13230,concepts=1323,
            fixed_practical_forward_gate_passed=passed,generation_operator_exported=passed,outer_folds=fold_results,
            pooled=dict(means={key:float(value.mean()) for key,value in tables.items()},concept_bootstrap=interval,pooled_CI_gate_passed=pooled_gate(interval)),
            alpha=.25,additional_subject_EEG_at_inference=False,no_query_target_access=True,outer_held_used_for_selection=False,
            outer_held_used_for_hyperparameter_selection=False,outer_held_used_for_fixed_generation_gate=True,
            latent_scaling_factor=.13025,latent_scale_already_applied_once=True,latent_flatten_order=FLATTEN,
            latent_receipt=ref(args.latent_receipt),latent_bank_sha256=LATENT_SHA,
            prediction_index=str(index.resolve()),prediction_index_sha256=sha(index),oof_coverage_min=int(coverage.min()),oof_coverage_max=int(coverage.max()),
            forward_folds=exported,query_measurement_coordinates='frozen D04 group-A whitened neural consensus16; predicted from query sub01 EEG only',
            prediction_comparison_means={key:float(value.mean()) for key,value in prediction_comparison.items()},
            query_measurement_is_predicted_not_directly_observed=True,per_query=ref(per_query),candidate_spec=ref(args.spec),
            provenance=ref(args.output/'provenance.json'),seconds=time.monotonic()-invoked,
            peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=budget.peak_GPU_bytes,
            no_adaptive_rank_lambda_or_alpha_sweep=True,SOTA_established=False,independent_confirmation=False)
        if passed:summary.update(query_measurements_path=str(query_path.resolve()),query_measurements_sha256=sha(query_path),query_measurements_shape=[13230,16])
        atomic(args.output/'summary.json',summary)
        atomic(args.output/'completion.json',dict(status=summary['status'],schema=SCHEMA,summary=ref(args.output/'summary.json')))
        log(dict(stage='D06_FORWARD_GATE_COMPLETE',status=summary['status'],fixed_practical_forward_gate_passed=passed))
        return 0
    except BaseException as error:
        atomic(args.output/'failure.json',dict(status='INCOMPLETE_NEURAL_LATENT_FORWARD_NO_GENERATION',error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-invoked))
        raise


if __name__=='__main__':raise SystemExit(main())
