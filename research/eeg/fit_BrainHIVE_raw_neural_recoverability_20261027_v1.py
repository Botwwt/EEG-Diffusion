"""D04: raw-neural cross-brain recoverability, with no visual target or generator.

sub01 predicts group A (02--05); independent group B (06--10) measures held
stimulus recovery only.  Every PCA/MCCA/normalizer/alignment sees outer FIT
stimuli only.  All ranks, ridges and practical gates are fixed before fitting.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import torch.nn.functional as F
from run_BrainHIVE_trialset_fastscreen_20261027_v1 import read_index

ROOT = Path(__file__).resolve().parent
SCHEMA = 'BrainHIVE_raw_neural_recoverability_20261027_v1'
GROUP_A = tuple(f'sub-{i:02d}' for i in range(2, 6))
GROUP_B = tuple(f'sub-{i:02d}' for i in range(6, 11))
SUBJECTS = ('sub-01',) + GROUP_A + GROUP_B
SEED = 1729
RANK = 16
RIDGE = .1


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.pending')
    assert not path.exists() and not temporary.exists()
    with temporary.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.rename(path)


def inverse_sqrt(matrix):
    eigenvalues, eigenvectors = torch.linalg.eigh((matrix + matrix.T) / 2)
    assert eigenvalues.min() > 0, 'Registered regularization must yield SPD covariance'
    return (eigenvectors * eigenvalues.rsqrt()[None]) @ eigenvectors.T


def fit_subject(values, dimensions=64):
    center = values.mean(0)
    scale = values.std(0, unbiased=False).clamp_min(1e-4)
    standardized = (values - center) / scale
    pca_center = standardized.mean(0)
    standardized = standardized - pca_center
    torch.manual_seed(SEED)
    _, singular, basis = torch.pca_lowrank(standardized, q=min(dimensions + 8, *standardized.shape), center=False, niter=2)
    basis = basis[:, :dimensions]
    projected = standardized @ basis
    pc_center = projected.mean(0)
    pc_scale = projected.std(0, unbiased=True).clamp_min(1e-4)
    return dict(center=center, scale=scale, PCA_center=pca_center, PCA_basis=basis,
        PC_center=pc_center, PC_scale=pc_scale,
        PCA_explained_variance=float(singular[:dimensions].square().sum() / standardized.square().sum().clamp_min(1e-12)))


def subject_transform(values, fitted):
    return (((((values - fitted['center']) / fitted['scale']) - fitted['PCA_center']) @ fitted['PCA_basis'])
             - fitted['PC_center']) / fitted['PC_scale']


def fit_mcca(views, rank=RANK):
    """Regularized SUMCOR generalized eigenproblem, diagonal cross blocks zero."""
    views = [value.double() for value in views]
    n, d = views[0].shape
    assert all(value.shape == (n, d) for value in views) and rank <= d
    centers = [value.mean(0) for value in views]
    x = [value - center for value, center in zip(views, centers)]
    whiten = [inverse_sqrt(value.T @ value / (n-1) + RIDGE * torch.eye(d, dtype=value.dtype, device=value.device)) for value in x]
    matrix = torch.zeros((len(views)*d, len(views)*d), dtype=x[0].dtype, device=x[0].device)
    for i in range(len(views)):
        for j in range(i+1, len(views)):
            cross = whiten[i] @ (x[i].T @ x[j] / (n-1)) @ whiten[j]
            matrix[i*d:(i+1)*d, j*d:(j+1)*d] = cross
            matrix[j*d:(j+1)*d, i*d:(i+1)*d] = cross.T
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    common = eigenvectors[:, -rank:].flip(1)
    projections = [whiten[i] @ common[i*d:(i+1)*d] for i in range(len(views))]
    canonical = [value @ projection for value, projection in zip(x, projections)]
    canonical_centers = [value.mean(0) for value in canonical]
    canonical_scales = [value.std(0, unbiased=True).clamp_min(1e-6) for value in canonical]
    unit = [(value-center)/scale for value, center, scale in zip(canonical, canonical_centers, canonical_scales)]
    group = torch.stack(unit).mean(0)
    group_center = group.mean(0)
    group_scale = group.std(0, unbiased=True).clamp_min(1e-6)
    group_z = (group-group_center)/group_scale
    covariance = group_z.T @ group_z / (n-1)
    group_whitener = inverse_sqrt(.9*covariance + .1*torch.eye(rank, dtype=group.dtype, device=group.device))
    return dict(view_centers=centers, projections=projections, canonical_centers=canonical_centers,
        canonical_scales=canonical_scales, group_center=group_center, group_scale=group_scale,
        group_whitener=group_whitener, selected_eigenvalues=eigenvalues[-rank:].flip(0),
        projection_weight_Frobenius=[float(value.norm()) for value in projections],
        rank=rank, regularization=RIDGE, group_covariance_shrinkage=.1)


def group_transform(views, fitted):
    individual = [((value.double()-center) @ projection-canonical_center)/canonical_scale
        for value, center, projection, canonical_center, canonical_scale in zip(views,
            fitted['view_centers'], fitted['projections'], fitted['canonical_centers'], fitted['canonical_scales'])]
    group = torch.stack(individual).mean(0)
    return ((group-fitted['group_center'])/fitted['group_scale']) @ fitted['group_whitener'], individual


def procrustes(a, b):
    u, _, vh = torch.linalg.svd(a.T @ b, full_matrices=False)
    return u @ vh


def ridge_fit(features, target):
    features, target = features.double(), target.double()
    center_x, center_y = features.mean(0), target.mean(0)
    x, y = features-center_x, target-center_y
    gram = x.T @ x / len(x) + RIDGE * torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    weights = torch.cholesky_solve(x.T @ y / len(x), torch.linalg.cholesky(gram))
    return dict(weights=weights, feature_center=center_x, target_center=center_y)


def ridge_predict(features, fitted):
    return (features.double()-fitted['feature_center']) @ fitted['weights'] + fitted['target_center']


def paired_metrics(prediction, measurement, items):
    prediction = F.normalize(prediction.double(), dim=1)
    measurement = F.normalize(measurement.double(), dim=1)
    groups = {}
    for i, item in enumerate(items):
        groups.setdefault(item['concept_id'], []).append(i)
    result = {key: np.empty(len(items), np.float64) for key in ('correct_cosine','sameconcept_deranged_cosine',
        'paired_cosine_gap','sameconcept10_top1','deranged_sameconcept10_top1','paired_top1_gap','sameconcept10_margin')}
    for concept, positions in groups.items():
        assert len(positions) == 10
        ids = torch.tensor(positions, device=prediction.device)
        similarity = (prediction[ids] @ measurement[ids].T).cpu().numpy()
        offset = 1 + int(hashlib.sha256(f'{SEED}|raw-neural-derangement|{concept}'.encode()).hexdigest()[:8],16) % 9
        perm = (np.arange(10)+offset) % 10
        correct = similarity.diagonal().copy()
        wrong = similarity[perm, np.arange(10)]
        top = (similarity.argmax(1) == np.arange(10)).astype(np.float64)
        null_top = (similarity[perm].argmax(1) == np.arange(10)).astype(np.float64)
        np.fill_diagonal(similarity, -np.inf)
        for key, value in dict(correct_cosine=correct, sameconcept_deranged_cosine=wrong,
            paired_cosine_gap=correct-wrong, sameconcept10_top1=top, deranged_sameconcept10_top1=null_top,
            paired_top1_gap=top-null_top, sameconcept10_margin=correct-similarity.max(1)).items():
            result[key][positions] = value
    return result


def correlations(a, b):
    a, b = a-a.mean(0), b-b.mean(0)
    return ((a*b).sum(0)/(a.square().sum(0).sqrt()*b.square().sum(0).sqrt()).clamp_min(1e-12)).cpu().tolist()


def bootstrap(items, values):
    names = sorted({row['concept_id'] for row in items})
    grouped = {name: [] for name in names}
    for i, row in enumerate(items):
        grouped[row['concept_id']].append(i)
    group_values = np.asarray([[values[key][grouped[name]].mean() for key in ('paired_cosine_gap','paired_top1_gap')] for name in names])
    sample = np.random.default_rng(SEED).integers(0, len(names), size=(1000,len(names)))
    draws = group_values[sample].mean(1)
    return dict(draws=1000, seed=SEED, unit='concept (ten paired stimuli)', descriptive_only=True,
        paired_cosine_gap95=np.quantile(draws[:,0],[.025,.975]).tolist(),
        paired_top1_gap95=np.quantile(draws[:,1],[.025,.975]).tolist())


def cpu_value(value):
    if torch.is_tensor(value):
        return value.cpu()
    if isinstance(value, dict):
        return {key: cpu_value(item) for key,item in value.items()}
    if isinstance(value, list):
        return [cpu_value(item) for item in value]
    return value


def self_test():
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    n, d, rank = 500, 8, 3
    latent = torch.randn(n,rank,dtype=torch.float64)
    a = [latent @ torch.randn(rank,d,dtype=torch.float64)+.01*torch.randn(n,d,dtype=torch.float64) for _ in range(4)]
    b = [latent @ torch.randn(rank,d,dtype=torch.float64)+.01*torch.randn(n,d,dtype=torch.float64) for _ in range(5)]
    fit = slice(0,400)
    model_a, model_b = fit_mcca([x[fit] for x in a],rank),fit_mcca([x[fit] for x in b],rank)
    centered_a = [x[fit]-x[fit].mean(0) for x in a]
    within = torch.block_diag(*[x.T@x/399+RIDGE*torch.eye(d,dtype=x.dtype) for x in centered_a])
    off = torch.cat([torch.cat([centered_a[i].T@centered_a[j]/399 if i!=j else torch.zeros((d,d),dtype=torch.float64)
                              for j in range(4)],dim=1) for i in range(4)],dim=0)
    projections = torch.cat(model_a['projections'],dim=0)
    residual = off@projections-within@projections@torch.diag(model_a['selected_eigenvalues'])
    assert residual.norm()/(off@projections).norm() < 1e-10
    af,_ = group_transform([x[fit] for x in a],model_a)
    bf,_ = group_transform([x[fit] for x in b],model_b)
    ah,_ = group_transform([x[400:] for x in a],model_a)
    bh,_ = group_transform([x[400:] for x in b],model_b)
    rotation = procrustes(af,bf)
    assert torch.allclose(rotation.T@rotation,torch.eye(rank,dtype=torch.float64),atol=1e-10)
    assert F.cosine_similarity(ah@rotation,bh).mean() > .98
    estimator = ridge_fit(latent[:400],af)
    assert F.cosine_similarity(ridge_predict(latent[400:],estimator)@rotation,bh).mean() > .98
    items = [dict(concept_id=f'c{i//10}') for i in range(100)]
    scores = paired_metrics(ah@rotation,bh,items)
    assert scores['sameconcept10_top1'].mean() > .95 and scores['paired_cosine_gap'].mean() > .5
    constant = paired_metrics(torch.ones_like(ah),bh,items)
    assert np.all(constant['paired_cosine_gap']==0) and np.all(constant['paired_top1_gap']==0)
    assert abs(constant['sameconcept10_top1'].mean()-.1)<1e-12
    q,_ = torch.linalg.qr(torch.randn(3,3,dtype=torch.float64))
    assert torch.allclose(procrustes(latent,latent@q),q,atol=1e-10)
    whitening = inverse_sqrt(torch.tensor([[2.,.3],[.3,1.]],dtype=torch.float64))
    assert torch.allclose(whitening@torch.tensor([[2.,.3],[.3,1.]],dtype=torch.float64)@whitening,torch.eye(2,dtype=torch.float64),atol=1e-10)
    fitted = fit_subject(torch.randn(80,12),dimensions=8)
    assert subject_transform(torch.randn(7,12),fitted).shape == (7,8)
    interval = bootstrap(items,scores)
    assert interval['paired_cosine_gap95'][0] > 0 and interval['paired_top1_gap95'][0] > 0
    print(json.dumps(dict(status='RAW_NEURAL_MCCA_SYNTHETIC_MATH_PASSED',checks=10,
        CUDA_initialized=torch.cuda.is_initialized(),real_data_read=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','output','spec'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--gpu-job-name')
    parser.add_argument('--max-seconds',type=float,default=600)
    parser.add_argument('--self-test',action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    assert all((args.inputs,args.output,args.spec,args.gpu_job_name)) and args.device=='cuda:0' and 0<args.max_seconds<=600
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    invoked=time.monotonic();stopped=[False]
    signal.signal(signal.SIGTERM,lambda number,frame:stopped.__setitem__(0,True))
    import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
    D.torch=torch
    auth=ROOT/'configs/resource_authorization_20261020_v3.json'
    document=dict(gpu_job_name=args.gpu_job_name,maximum_seconds=args.max_seconds,
        maximum_extra_CPU_bytes=10<<30,maximum_GPU_bytes=20<<30,minimum_CPU_headroom_bytes=10<<30,
        maximum_output_bytes=128<<20,resource_authorization=ref(auth))
    wrapper=D.verify_wrapper(document);budget=D.Budget(document)
    def check():
        assert not stopped[0] and time.monotonic()-invoked<args.max_seconds,'Finite raw-neural screen limit'
        budget.check()
    def log(value):
        value=value|dict(seconds=time.monotonic()-invoked)
        with (args.output/'progress.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(value,allow_nan=False)+'\n');stream.flush()
        print(json.dumps(value),flush=True)
    try:
        torch.set_num_threads(2);torch.manual_seed(SEED);torch.cuda.set_device(torch.device('cuda:0'))
        torch.cuda.reset_peak_memory_stats();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        inputs=json.loads(args.inputs.read_bytes());spec=json.loads(args.spec.read_bytes())
        assert inputs['status']=='COMPLETE_RAW_NEURAL_T17_ALL10_INPUTS' and inputs['schema']==spec['schema']==SCHEMA
        assert set(inputs['subjects'])==set(SUBJECTS) and inputs['rows']==13230 and inputs['concepts']==1323
        assert spec['query_subject']=='sub-01' and spec['group_A']==list(GROUP_A) and spec['group_B']==list(GROUP_B)
        assert spec['practical_gate']==dict(per_outer_fold_paired_cosine_gap_minimum=.01,
            per_outer_fold_sameconcept10_top1_minimum=.12,pooled_paired_cosine_gap_CI_lower_positive=True,
            pooled_paired_top1_gap_CI_lower_positive=True,both_groups_both_outer_folds_required=True)
        index=Path(inputs['prediction_index']);assert sha(index)==inputs['prediction_index_sha256']
        control_bindings={key:ref(path) for key,path in (('inputs',args.inputs),('spec',args.spec),('index',index))}
        items=read_index(index);assert len(items)==13230 and [row['row'] for row in items]==list(range(13230))
        source_bindings={name:dict(path=inputs['subjects'][name]['path'],sha256=inputs['subjects'][name]['sha256']) for name in SUBJECTS}
        for name,binding in source_bindings.items():
            check();assert sha(binding['path'])==binding['sha256']
        source_digest=sha(__file__)
        atomic(args.output/'provenance.json',dict(schema=SCHEMA,inputs=ref(args.inputs),spec=ref(args.spec),source=ref(__file__),
            source_bindings=source_bindings,wrapper=wrapper,query_subject='sub-01',group_A=list(GROUP_A),group_B=list(GROUP_B),
            visual_F_or_latent_loaded=False,held_B_used_only_for_evaluation=True,query_in_teacher_consensus=False,
            fixed_PCA_rank=64,fixed_MCCA_rank=16,MCCA_within_ridge=.1,query_ridge=.1,
            source_preprocessing_MVNN_train_only_established=False,independent_session_metadata_available=False,
            limitation='shared, linearly recoverable 17-channel R4 components only; private/nonlinear neural content remains untested'))
        per_fold=[];tables={case:{key:np.empty(13230,np.float64) for key in ('correct_cosine','sameconcept_deranged_cosine',
            'paired_cosine_gap','sameconcept10_top1','deranged_sameconcept10_top1','paired_top1_gap','sameconcept10_margin')}
            for case in ('groupA_to_groupB','query01_to_groupB')}
        query_predictions=np.empty((13230,16),np.float32);coverage=np.zeros(13230,np.int32)
        for fold in (0,1):
            fit_positions=[i for i,row in enumerate(items) if row['fold']!=fold]
            held_positions=[i for i,row in enumerate(items) if row['fold']==fold]
            fit=torch.tensor(fit_positions,device='cuda');held=torch.tensor(held_positions,device='cuda')
            held_items=[items[i] for i in held_positions]
            scores={};subject_models={};subject_diagnostics={}
            for name in SUBJECTS:
                check();array=np.load(source_bindings[name]['path'],mmap_mode='r',allow_pickle=False)
                assert array.shape==(13230,17,250) and array.dtype==np.float32
                values=torch.from_numpy(np.array(array,copy=True).reshape(13230,-1)).cuda();del array
                assert torch.isfinite(values).all()
                model=fit_subject(values[fit]);scores[name]=subject_transform(values,model).double()
                subject_models[name]=model
                subject_diagnostics[name]=dict(PCA_explained_variance=model['PCA_explained_variance'])
                del values
                log(dict(stage='OUTER_TRAIN_PCA64_COMPLETE',outer_fold=fold,subject=name,
                    train_rows=len(fit),held_rows=len(held),**subject_diagnostics[name]))
            model_A=fit_mcca([scores[name][fit] for name in GROUP_A])
            model_B=fit_mcca([scores[name][fit] for name in GROUP_B])
            A_fit,_=group_transform([scores[name][fit] for name in GROUP_A],model_A)
            B_fit,_=group_transform([scores[name][fit] for name in GROUP_B],model_B)
            rotation=procrustes(A_fit,B_fit)
            query_model=ridge_fit(scores['sub-01'][fit],A_fit)
            predicted_A=ridge_predict(scores['sub-01'][held],query_model)
            # B-held neural data enter here, after every fitted state is frozen.
            A_held,individual_A=group_transform([scores[name][held] for name in GROUP_A],model_A)
            B_held,individual_B=group_transform([scores[name][held] for name in GROUP_B],model_B)
            group_case=paired_metrics(A_held@rotation,B_held,held_items)
            query_case=paired_metrics(predicted_A@rotation,B_held,held_items)
            for case,values in (('groupA_to_groupB',group_case),('query01_to_groupB',query_case)):
                for key,value in values.items():tables[case][key][held_positions]=value
            query_predictions[held_positions]=predicted_A.float().cpu().numpy();coverage[held_positions]+=1
            diagnostics={}
            for names,model,individual in ((GROUP_A,model_A,individual_A),(GROUP_B,model_B,individual_B)):
                for j,name in enumerate(names):
                    diagnostics[name]=dict(projection_weight_Frobenius=model['projection_weight_Frobenius'][j],
                        pre_standardization_train_canonical_std=model['canonical_scales'][j].cpu().tolist(),
                        held_standardized_canonical_RMS=individual[j].square().mean(0).sqrt().cpu().tolist())
            result=dict(outer_fold=fold,fit_rows=len(fit),held_rows=len(held),subject_PCA=subject_diagnostics,
                subject_canonical_diagnostics=diagnostics,A_MCCA_eigenvalues=model_A['selected_eigenvalues'].cpu().tolist(),
                B_MCCA_eigenvalues=model_B['selected_eigenvalues'].cpu().tolist(),
                held_A_aligned_B_correlations=correlations(A_held@rotation,B_held),
                held_query_aligned_B_correlations=correlations(predicted_A@rotation,B_held),cases={},
                training_white_covariance_or_correlations_are_not_success_evidence=True)
            for case,values in (('groupA_to_groupB',group_case),('query01_to_groupB',query_case)):
                means={key:float(value.mean()) for key,value in values.items()}
                means['practical_fold_gate_passed']=(means['paired_cosine_gap']>.01 and means['sameconcept10_top1']>=.12)
                result['cases'][case]=means
            per_fold.append(result);atomic(args.output/f'fold{fold}_summary.json',result)
            torch.save(cpu_value(dict(subject_models=subject_models,MCCA_A=model_A,MCCA_B=model_B,
                A_to_B_rotation=rotation,query_to_A=query_model,outer_fold=fold,
                fit_source_rows=[items[i]['source_row'] for i in fit_positions],
                held_source_rows=[items[i]['source_row'] for i in held_positions])),args.output/f'fold{fold}_models.pt')
            log(dict(stage='OUTER_FOLD_RAW_NEURAL_COMPLETE',**result))
            del scores,subject_models,model_A,model_B,A_fit,B_fit,A_held,B_held,predicted_A
        assert np.all(coverage==1)
        per_query=args.output/'per_query.jsonl.gz'
        with gzip.open(per_query,'wt',encoding='utf-8') as stream:
            for i,item in enumerate(items):
                stream.write(json.dumps(item|dict(cases={case:{key:float(value[i]) for key,value in table.items()} for case,table in tables.items()}))+'\n')
        np.save(args.output/'query01_prediction_of_groupA.npy',query_predictions,allow_pickle=False)
        pooled={}
        for case,values in tables.items():
            interval=bootstrap(items,values)
            pooled[case]=dict(means={key:float(value.mean()) for key,value in values.items()},concept_bootstrap=interval,
                pooled_CI_gate_passed=(interval['paired_cosine_gap95'][0]>0 and interval['paired_top1_gap95'][0]>0))
        passed=all(fold['cases'][case]['practical_fold_gate_passed'] for fold in per_fold for case in tables)
        passed=passed and all(value['pooled_CI_gate_passed'] for value in pooled.values())
        check();assert sha(__file__)==source_digest
        for binding in control_bindings.values():assert sha(binding['path'])==binding['sha256']
        for binding in source_bindings.values():assert sha(binding['path'])==binding['sha256']
        assert sum(path.stat().st_size for path in args.output.rglob('*') if path.is_file())<128<<20
        summary=dict(status='COMPLETE_RAW_NEURAL_RECOVERABILITY_SCREEN',schema=SCHEMA,rows=13230,concepts=1323,
            query_subject='sub-01',group_A=list(GROUP_A),group_B=list(GROUP_B),
            outer_folds=per_fold,pooled=pooled,fixed_practical_screen_gate_passed=passed,
            visual_bridge_or_generation_authorized_by_this_script=False,
            next_decision='Evidence supports considering a separately registered raw-EEG denoising/calibration question' if passed else 'STOP: no visual bridge or generation from this direction',
            all_rows_retained=True,held_B_evaluation_only=True,visual_F_or_latent_loaded=False,
            query_subject_in_consensus=False,private_neural_content_excluded=False,
            nonlinear_recoverability_tested=False,session_or_display_artifacts_excluded=False,
            independent_confirmation=False,SOTA_established=False,per_query=ref(per_query),
            query_predictions=ref(args.output/'query01_prediction_of_groupA.npy'),
            seconds=time.monotonic()-invoked,peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=budget.peak_GPU_bytes)
        atomic(args.output/'summary.json',summary)
        atomic(args.output/'completion.json',dict(status=summary['status'],schema=SCHEMA,summary=ref(args.output/'summary.json')))
        print(json.dumps(summary),flush=True)
        return 0
    except BaseException as error:
        atomic(args.output/'failure.json',dict(status='INCOMPLETE_RAW_NEURAL_SCREEN_NO_ADAPTIVE_REPLAY',error=repr(error),
            traceback=traceback.format_exc(),expected_rows=13230,seconds=time.monotonic()-invoked))
        raise


if __name__=='__main__':
    raise SystemExit(main())
