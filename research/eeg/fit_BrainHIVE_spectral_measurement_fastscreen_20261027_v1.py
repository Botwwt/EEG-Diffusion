"""Direction03: nested concept-fold readouts of 64 native VAE measurements.

Both c_only and raw_eeg use a train-only standardized PCA64/ridge64 readout.
No outer-held target determines PCA, ridge strength, or measurement weight.
Inner-CV skill becomes a finite proximal heuristic, NOT a Kalman precision.
"""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from run_BrainHIVE_trialset_fastscreen_20261027_v1 import read_index

ROOT = Path(__file__).resolve().parent
SCHEMA = 'BrainHIVE_spectral_measurement_fastscreen_20261027_v1'
ARMS = ('c_only', 'raw_eeg')
LAMBDAS = (.01, .1, 1., 10., 100.)
SEED = 1729
DIM = 64
LATENT_RECEIPT_SHA256 = 'a6fb379d25fe3738c47c036fa591c0603f952ba9e119763c487151462a4dae6f'
LATENT_SHA256 = 'eda12be7eeba9308437f9d58e5809f198dcc0a730fea8d63b2cd7158f88d0039'
COEFFICIENT_ORDER = 'channel0..3,ky0..3,kx0..3; C-order flatten'
BASIS_DEFINITION = 'B[k,n]=alpha(k)*cos(pi*(n+0.5)*k/64); alpha(0)=sqrt(1/64); alpha(k>0)=sqrt(2/64); k=0..3,n=0..63'
COORDINATES = 'orthonormal 4x4 DCT-II coefficients per channel of scaled native VAE MODE latent; order channel,ky,kx'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def atomic(path, value):
    path = Path(path)
    pending = path.with_name(path.name + '.pending')
    assert not path.exists() and not pending.exists()
    with pending.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    pending.rename(path)


def save_npy(path, value):
    path = Path(path)
    assert not path.exists()
    with path.open('xb') as stream:
        np.save(stream, np.asarray(value), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def dct_basis(device='cpu', dtype=torch.float64):
    n = torch.arange(64, device=device, dtype=dtype)
    k = torch.arange(4, device=device, dtype=dtype)[:, None]
    scale = torch.full((4, 1), math.sqrt(2 / 64), device=device, dtype=dtype)
    scale[0] = math.sqrt(1 / 64)
    return scale * torch.cos(math.pi * (n[None] + .5) * k / 64)


def measure(latent, basis):
    assert latent.ndim == 4 and latent.shape[1:] == (4, 64, 64)
    return torch.einsum('ah,nchw,bw->ncab', basis, latent, basis).reshape(len(latent), 64)


def synthesize(coefficients, basis):
    return torch.einsum('ah,ncab,bw->nchw', basis, coefficients.reshape(-1, 4, 4, 4), basis)


def preprocess_fit(features, dimensions=DIM):
    center = features.mean(0)
    scale = features.std(0, unbiased=False).clamp_min(1e-4)
    standardized = (features - center) / scale
    pca_center = standardized.mean(0)
    centered = standardized - pca_center
    torch.manual_seed(SEED)
    _, singular, vectors = torch.pca_lowrank(centered, q=min(dimensions + 8, *centered.shape), center=False, niter=2)
    basis = vectors[:, :dimensions]
    assert basis.shape[1] == dimensions
    explained = float(singular[:dimensions].square().sum() / centered.square().sum().clamp_min(1e-12))
    return dict(center=center, scale=scale, PCA_center=pca_center, PCA_basis=basis,
                PCA_explained_variance=explained)


def transform(features, fitted):
    return (((features - fitted['center']) / fitted['scale']) - fitted['PCA_center']) @ fitted['PCA_basis']


def ridge_fit(features, targets, regularization):
    features, targets = features.double(), targets.double()
    mx, my = features.mean(0), targets.mean(0)
    target_scale = targets.std(0, unbiased=False).clamp_min(1e-4)
    x, y = features - mx, (targets - my) / target_scale
    gram = x.T @ x / len(x) + regularization * torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
    weights = torch.cholesky_solve(x.T @ y / len(x), torch.linalg.cholesky(gram))
    return dict(feature_center=mx, target_center=my, target_scale=target_scale, weights=weights)


def predict(features, model):
    return ((features.double() - model['feature_center']) @ model['weights']) * model['target_scale'] + model['target_center']


def inner_folds(items):
    names = sorted({row['concept_id'] for row in items},
                   key=lambda name: (hashlib.sha256(f'{SEED}|spectral-inner|{name}'.encode()).hexdigest(), name))
    assignment = {name: position % 5 for position, name in enumerate(names)}
    return [assignment[row['concept_id']] for row in items]


def fit_readout(fit_features, fit_targets, held_features, fit_items, check, event, *, dimensions=DIM):
    """No held target argument exists; it cannot affect lambda/beta/model fit."""
    labels = torch.tensor(inner_folds(fit_items), device=fit_features.device)
    cv_predictions = torch.empty((len(LAMBDAS), len(fit_targets), 64), device=fit_features.device, dtype=torch.float64)
    baseline_predictions = torch.empty_like(fit_targets, dtype=torch.float64)
    normalized_scores = torch.zeros(len(LAMBDAS), dtype=torch.float64, device=fit_features.device)
    cv_coverage = torch.zeros(len(fit_targets), dtype=torch.int32, device=fit_features.device)
    for inner in range(5):
        check()
        validation = labels == inner
        training = ~validation
        assert training.any() and validation.any()
        fitted = preprocess_fit(fit_features[training], dimensions)
        train_x = transform(fit_features[training], fitted)
        valid_x = transform(fit_features[validation], fitted)
        mu = fit_targets[training].double().mean(0)
        target_scale = fit_targets[training].double().std(0, unbiased=False).clamp_min(1e-4)
        baseline_predictions[validation] = mu
        scores = []
        for index, penalty in enumerate(LAMBDAS):
            model = ridge_fit(train_x, fit_targets[training], penalty)
            value = predict(valid_x, model)
            cv_predictions[index, validation] = value
            score = ((value - fit_targets[validation]) / target_scale).square().sum()
            normalized_scores[index] += score
            scores.append(float(score / (int(validation.sum()) * 64)))
        cv_coverage[validation] += 1
        event(dict(stage='INNER_READOUT_CV', inner_fold=inner, training_rows=int(training.sum()),
            validation_rows=int(validation.sum()), lambdas=list(LAMBDAS), normalized_MSE=scores,
            PCA_explained_variance=fitted['PCA_explained_variance'], preprocessing_refit_inside_inner_fold=True))
    assert torch.all(cv_coverage == 1)
    scores = normalized_scores / (len(fit_targets) * 64)
    # Stronger regularization wins an exact numerical tie.
    selected = min(range(len(LAMBDAS)), key=lambda i: (float(scores[i]), -LAMBDAS[i]))
    errors = (cv_predictions[selected] - fit_targets).square().mean(0)
    base_errors = (baseline_predictions - fit_targets).square().mean(0)
    beta = torch.where(base_errors > 1e-12, (1 - errors / base_errors.clamp_min(1e-12)).clamp(0, 1),
                       torch.zeros_like(base_errors))
    fitted = preprocess_fit(fit_features, dimensions)
    model = ridge_fit(transform(fit_features, fitted), fit_targets, LAMBDAS[selected])
    predictions = predict(transform(held_features, fitted), model)
    receipt = dict(selected_lambda=LAMBDAS[selected], inner_lambda_normalized_MSE=scores.cpu().tolist(),
        inner_prediction_MSE_per_coefficient=errors.cpu().tolist(),
        inner_constant_mean_MSE_per_coefficient=base_errors.cpu().tolist(), beta=beta.cpu().tolist(),
        positive_beta_coefficients=int((beta > 0).sum()), beta_mean=float(beta.mean()),
        beta_rule='clip(1 - innerOOF_rawMSE / innerOOF_innerTrainMean_rawMSE, 0, 1); zero if baseline variance <=1e-12',
        reliability_is_proximal_heuristic=True, independent_reliability_calibration=False,
        lambda_and_beta_share_inner_CV_predictions=True, outer_held_used_for_selection=False,
        inner_preprocessing_refit=True, readout_PCA_explained_variance=fitted['PCA_explained_variance'],
        readout_parameters=dimensions * 64 + 64, input_dimensions=fit_features.shape[1])
    return predictions, beta, receipt, fitted, model


def self_test():
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    basis = dct_basis()
    assert torch.allclose(basis @ basis.T, torch.eye(4, dtype=torch.float64), atol=2e-15)
    latent = torch.randn(2, 4, 64, 64, dtype=torch.float64)
    coefficients = measure(latent, basis)
    assert torch.allclose(measure(synthesize(coefficients, basis), basis), coefficients, atol=1e-13)
    low = synthesize(coefficients, basis)
    assert torch.allclose(synthesize(measure(low, basis), basis), low, atol=1e-13)
    dc = measure(torch.ones(1, 4, 64, 64, dtype=torch.float64), basis).reshape(1, 4, 4, 4)
    assert torch.allclose(dc[:, :, 0, 0], torch.full((1, 4), 64., dtype=torch.float64))
    dc[:, :, 0, 0] = 0
    assert dc.abs().max() < 1e-12
    assert torch.allclose(measure(latent * .13025, basis), measure(latent, basis) * .13025, atol=1e-13)
    n = 120
    features = torch.randn(n, 12)
    targets = features.double() @ torch.randn(12, 64, dtype=torch.float64)
    items = [dict(concept_id=f'c{i//10}') for i in range(n)]
    events = []
    prediction, beta, receipt, _, _ = fit_readout(features, targets, features[:7], items,
        lambda: None, events.append, dimensions=8)
    assert prediction.shape == (7, 64) and beta.shape == (64,) and torch.all((beta >= 0) & (beta <= 1))
    assert len(events) == 5 and receipt['selected_lambda'] in LAMBDAS
    assert all(event['preprocessing_refit_inside_inner_fold'] for event in events)
    labels = inner_folds(items)
    assert all(len(set(labels[i:i+10])) == 1 for i in range(0, n, 10))
    print(json.dumps(dict(status='SPECTRAL_MEASUREMENT_SYNTHETIC_CHECKS_PASSED', checks=9,
        CUDA_initialized=torch.cuda.is_initialized(), real_data_read=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('trials', 'index', 'native-parent-binding', 'latent-receipt', 'latents', 'output', 'candidate-spec'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-job-name')
    parser.add_argument('--max-seconds', type=float, default=600)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    assert all((args.trials, args.index, args.native_parent_binding, args.latent_receipt,
                args.output, args.gpu_job_name, args.candidate_spec))
    assert args.device == 'cuda:0' and 0 < args.max_seconds <= 600
    assert not args.output.exists(), 'Preserve every earlier attempt'
    args.output.mkdir(parents=True)
    invoked = time.monotonic()
    stopped = [False]
    signal.signal(signal.SIGTERM, lambda number, frame: stopped.__setitem__(0, True))
    import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
    D.torch = torch
    authorization = ROOT / 'configs/resource_authorization_20261020_v3.json'
    document = dict(gpu_job_name=args.gpu_job_name, maximum_seconds=args.max_seconds,
        maximum_extra_CPU_bytes=10 << 30, maximum_GPU_bytes=20 << 30,
        minimum_CPU_headroom_bytes=10 << 30, maximum_output_bytes=128 << 20,
        resource_authorization=ref(authorization))
    wrapper = D.verify_wrapper(document)
    budget = D.Budget(document)
    def check():
        assert not stopped[0] and time.monotonic() - invoked < args.max_seconds, 'Finite spectral fit stop'
        budget.check()
    def log(value):
        value = value | dict(seconds=time.monotonic() - invoked)
        with (args.output / 'training.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n')
            stream.flush()
        print(json.dumps(value), flush=True)
    try:
        torch.set_num_threads(2)
        torch.manual_seed(SEED)
        torch.cuda.set_device(torch.device('cuda:0'))
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        assert sha(args.latent_receipt) == LATENT_RECEIPT_SHA256
        latent_receipt = json.loads(args.latent_receipt.read_bytes())
        assert latent_receipt['status'] == 'COMPLETE_ALL13230_ALLOWED_T_NATIVE_VAE_MODE_LATENTS'
        assert latent_receipt['shape'] == [13230, 4, 64, 64] and latent_receipt['dtype'] == 'float32'
        assert latent_receipt['sha256'] == LATENT_SHA256
        assert latent_receipt['registration']['VAE_scaling_factor'] == .13025
        latent_path = args.latents or Path(latent_receipt['path'])
        assert sha(latent_path) == LATENT_SHA256
        native = json.loads(args.native_parent_binding.read_bytes())
        assert native['status'] == 'COMPLETE_NATIVE_PARENT_OOF_PREDICTIONS'
        assert native['predictions_shape'] == [13230, 1024] and native['oof_coverage_min'] == native['oof_coverage_max'] == 1
        assert native['prediction_index_sha256'] == sha(args.index)
        assert sha(native['predictions_path']) == native['predictions_sha256']
        paths = dict(trials=args.trials, index=args.index, native_parent_binding=args.native_parent_binding,
                     native_conditions=Path(native['predictions_path']), latent_receipt=args.latent_receipt,
                     latent_bank=latent_path, source=Path(__file__), candidate_spec=args.candidate_spec,
                     helper_source=ROOT / 'run_BrainHIVE_trialset_fastscreen_20261027_v1.py')
        bindings = {key: ref(path) for key, path in paths.items()}
        spec = json.loads(args.candidate_spec.read_bytes())
        assert spec['schema'] == SCHEMA and spec['arms'] == list(ARMS)
        items = read_index(args.index)
        assert len(items) == 13230 and len({row['row'] for row in items}) == 13230
        assert len({row['subject_id'] for row in items}) == 1 and {row['fold'] for row in items} == {0, 1}
        grouped = {}
        for row in items:
            grouped.setdefault(row['concept_id'], []).append(row)
        assert len(grouped) == 1323 and all(len(g) == 10 and len({r['fold'] for r in g}) == 1 for g in grouped.values())
        source_rows = [int(row['source_row']) for row in items]
        assert set(source_rows) == set(latent_receipt['array_rows'])
        latent_lookup = {source: i for i, source in enumerate(latent_receipt['array_rows'])}
        trials = np.load(args.trials, mmap_mode='r', allow_pickle=False)
        bank = np.load(latent_path, mmap_mode='r', allow_pickle=False)
        condition_array = np.load(native['predictions_path'], mmap_mode='r', allow_pickle=False)
        assert trials.dtype == bank.dtype == condition_array.dtype == np.float32
        assert trials.shape == (13230, 4, 17, 250) and bank.shape == (13230, 4, 64, 64)
        assert condition_array.shape == (13230, 1024)
        basis = dct_basis(device='cuda')
        x_chunks, measurement_chunks = [], []
        for start in range(0, len(items), 256):
            check()
            batch = items[start:start+256]
            raw = np.array(trials[[row['row'] for row in batch]], copy=True).mean(1)
            x_chunks.append(torch.from_numpy(raw.reshape(len(batch), -1)).cuda())
            latent = torch.from_numpy(np.array(bank[[latent_lookup[int(row['source_row'])] for row in batch]], copy=True)).cuda().double()
            measurement_chunks.append(measure(latent, basis))
        raw_eeg = torch.cat(x_chunks)
        measurements = torch.cat(measurement_chunks)
        raw_c = torch.from_numpy(np.array(condition_array, copy=True)).cuda()
        del trials, bank, condition_array, x_chunks, measurement_chunks
        assert torch.isfinite(raw_eeg).all() and torch.isfinite(measurements).all() and torch.isfinite(raw_c).all()
        basis_path = args.output / 'DCT_basis.npy'
        save_npy(basis_path, basis.cpu().numpy())
        provenance = dict(schema=SCHEMA, input_bindings=bindings, wrapper=wrapper,
            coefficient_order=COEFFICIENT_ORDER, basis_definition=BASIS_DEFINITION,
            latent_scaling_factor=.13025, latent_scale_already_applied_once=True,
            target='verified native FP32 VAE posterior MODE * scaling_factor, then orthonormal DCT; no second scale',
            arms=list(ARMS), seed=SEED, lambda_grid=list(LAMBDAS), outer_held_used_for_selection=False,
            matched_readout='standardize -> PCA64 -> target-standardized ridge64, with identical hyperparameter rule',
            feature_preprocessing_refit_in_each_inner_fold=True,
            ridge_objective='mean target-standardized squared error + lambda*||W||^2',
            beta_is_proximal_heuristic=True, independent_reliability_calibration=False,
            c_only_inherits_existing_native_encoder_exposure_and_OOF_role_distribution=True,
            readout_fold_test_is_not_independent_whole_system_confirmation=True)
        atomic(args.output / 'provenance.json', provenance)
        outputs = {arm: np.empty((13230, 64), np.float32) for arm in ARMS}
        betas = {arm: np.empty((2, 64), np.float32) for arm in ARMS}
        coverage = np.zeros(13230, np.int32)
        fold_summaries, per_query = [], []
        for fold in (0, 1):
            fit_positions = [i for i, row in enumerate(items) if row['fold'] != fold]
            held_positions = [i for i, row in enumerate(items) if row['fold'] == fold]
            fit = torch.tensor(fit_positions, device='cuda')
            held = torch.tensor(held_positions, device='cuda')
            fit_items = [items[i] for i in fit_positions]
            mu = measurements[fit].mean(0)
            baseline_mse = (measurements[held] - mu).square().mean(0)
            fold_summary = dict(outer_fold=fold, fit_rows=len(fit), held_rows=len(held),
                               outer_train_mean_held_MSE_per_coefficient=baseline_mse.cpu().tolist(), arms={})
            coverage[held_positions] += 1
            for arm, features in (('c_only', raw_c), ('raw_eeg', raw_eeg)):
                check()
                predicted, beta, result, preprocessing, model = fit_readout(
                    features[fit], measurements[fit], features[held], fit_items, check,
                    lambda value: log(dict(outer_fold=fold, arm=arm, **value)))
                actual_errors = (predicted - measurements[held]).square()
                mse = actual_errors.mean(0)
                pred_centered = predicted - predicted.mean(0)
                target_centered = measurements[held] - measurements[held].mean(0)
                correlation = (pred_centered * target_centered).sum(0) / (
                    pred_centered.square().sum(0).sqrt() * target_centered.square().sum(0).sqrt()).clamp_min(1e-12)
                result.update(held_raw_MSE=float(mse.mean()), held_MSE_per_coefficient=mse.cpu().tolist(),
                    held_skill_vs_outer_train_mean=(1 - mse / baseline_mse.clamp_min(1e-12)).cpu().tolist(),
                    held_correlation_per_coefficient=correlation.cpu().tolist(),
                    held_metrics_used_for_selection=False)
                outputs[arm][held_positions] = predicted.float().cpu().numpy()
                betas[arm][fold] = beta.float().cpu().numpy()
                fold_summary['arms'][arm] = result
                cpu_errors = actual_errors.cpu().numpy()
                for j, original in enumerate(held_positions):
                    per_query.append(items[original] | dict(arm=arm, raw_coefficient_MSE=float(cpu_errors[j].mean()),
                        raw_coefficient_squared_errors=cpu_errors[j].tolist(), outer_held_metric_not_used_for_selection=True))
                checkpoint = dict(preprocessing={key: value.cpu() if torch.is_tensor(value) else value for key, value in preprocessing.items()},
                    ridge={key: value.cpu() for key, value in model.items()}, beta=beta.cpu(),
                    schema=SCHEMA, arm=arm, outer_fold=fold, selected_lambda=result['selected_lambda'],
                    held_source_rows=[items[i]['source_row'] for i in held_positions],
                    fit_source_rows=[items[i]['source_row'] for i in fit_positions])
                torch.save(checkpoint, args.output / f'fold{fold}_{arm}.pt')
                log(dict(stage='OUTER_FOLD_FINAL_READOUT', outer_fold=fold, arm=arm,
                    selected_lambda=result['selected_lambda'], beta_mean=result['beta_mean'],
                    positive_beta_coefficients=result['positive_beta_coefficients'],
                    held_raw_MSE=result['held_raw_MSE'], outer_held_used_for_selection=False))
            fold_summaries.append(fold_summary)
            atomic(args.output / f'fold{fold}_summary.json', fold_summary)
        assert np.all(coverage == 1)
        index_out = args.output / 'prediction_index.jsonl'
        index_out.write_text(''.join(json.dumps(row) + '\n' for row in items), encoding='utf-8')
        with gzip.open(args.output / 'per_query.jsonl.gz', 'wt', encoding='utf-8') as stream:
            stream.write(''.join(json.dumps(row, allow_nan=False) + '\n' for row in per_query))
        summary = dict(status='COMPLETE_EEG_SPECTRAL_MEASUREMENT_SCREEN', schema=SCHEMA,
            rows_per_arm=13230, subject_id=items[0]['subject_id'], folds=fold_summaries, arms={},
            prediction_index=str(index_out.resolve()), prediction_index_sha256=sha(index_out),
            latent_scaling_factor=.13025, latent_scale_already_applied_once=True,
            latent_receipt=ref(args.latent_receipt), latent_bank_sha256=LATENT_SHA256,
            coefficient_order=COEFFICIENT_ORDER, basis_definition=BASIS_DEFINITION,
            basis_path=str(basis_path.resolve()), basis_sha256=sha(basis_path), basis_shape=[4, 64],
            prediction_coordinates=COORDINATES, lambda_grid=list(LAMBDAS), PCA_dimensions=64,
            outer_held_used_for_selection=False, beta_is_proximal_heuristic=True,
            independent_reliability_calibration=False, no_query_target_scaling=True,
            per_query=ref(args.output / 'per_query.jsonl.gz'), provenance=ref(args.output / 'provenance.json'),
            claim_limit='single-subject T-only readout screen; no image gain, independent confirmation, exact posterior or SOTA claim')
        for arm in ARMS:
            folder = args.output / arm
            folder.mkdir()
            path, beta_path = folder / 'measurements.npy', folder / 'beta.npy'
            assert np.isfinite(outputs[arm]).all() and np.isfinite(betas[arm]).all()
            save_npy(path, outputs[arm])
            save_npy(beta_path, betas[arm])
            summary['arms'][arm] = dict(measurements_path=str(path.resolve()), measurements_sha256=sha(path),
                measurements_shape=[13230, 64], prediction_coordinates=COORDINATES,
                beta_path=str(beta_path.resolve()), beta_sha256=sha(beta_path), beta_shape=[2, 64],
                beta_rule='clip(1 - innerOOF_rawMSE / innerOOF_innerTrainMean_rawMSE, 0, 1)',
                beta_frozen_by_outer_fold=True, oof_coverage_min=1, oof_coverage_max=1,
                held_raw_MSE=float(np.mean([row['raw_coefficient_MSE'] for row in per_query if row['arm'] == arm])))
        for key, path in paths.items():
            assert sha(path) == bindings[key]['sha256'], 'A bound source/input changed during fitting'
        check()
        assert sum(path.stat().st_size for path in args.output.rglob('*') if path.is_file()) < 128 << 20
        summary.update(seconds=time.monotonic() - invoked, peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,
                       peak_GPU_bytes=budget.peak_GPU_bytes)
        atomic(args.output / 'summary.json', summary)
        atomic(args.output / 'completion.json', dict(status=summary['status'], schema=SCHEMA,
            summary=ref(args.output / 'summary.json'), expected_rows=13230, observed_rows=13230))
        print(json.dumps(summary), flush=True)
        return 0
    except BaseException as error:
        atomic(args.output / 'failure.json', dict(status='FAILED_FIXED_DENOMINATOR_PRESERVED',
            error=repr(error), traceback=traceback.format_exc(), expected_rows=13230, seconds=time.monotonic() - invoked))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
