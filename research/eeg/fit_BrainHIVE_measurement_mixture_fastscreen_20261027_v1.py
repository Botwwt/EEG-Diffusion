"""Finite concept-OOF conditional Gaussian-mixture representation screen.

Outer-fit-only EEG/F coordinates; inner concept crossfit estimates B-prediction
residual covariance GIVEN those coordinates.  This is not full-pipeline OOF
uncertainty calibration, physical EEG noise, or an exact posterior for images.
No neural network, UNet fit, held-based selection, or query-target norm scale.
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
import torch.nn.functional as F

from run_BrainHIVE_trialset_fastscreen_20261027_v1 import read_index, validate_index, within_concept

ROOT = Path(__file__).resolve().parent
SCHEMA = 'BrainHIVE_measurement_mixture_fastscreen_20261027_v1'
ARMS = ('posterior_mean', 'component_mean')
COORDINATES = 'raw F-target coordinates by inverse outer-train centered PCA; no L2 renormalization or query-target scaling'
SEED = 1729
DIM = 64
COMPONENTS = 16
EM_STEPS = 30
RIDGE = .1
VAR_FLOOR = .05


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic(path, value):
    path = Path(path)
    pending = path.with_name(path.name + '.pending')
    assert not path.exists() and not pending.exists()
    with pending.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    pending.rename(path)


def save_npy(path, array):
    path = Path(path)
    assert not path.exists()
    with path.open('xb') as stream:
        np.save(stream, np.asarray(array), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def pca_fit(values, dimensions, *, randomized=False, seed=SEED):
    center = values.mean(0)
    centered = values - center
    if randomized:
        torch.manual_seed(seed)
        _, singular, vectors = torch.pca_lowrank(centered, q=min(dimensions + 8, *centered.shape), center=False, niter=4)
        basis = vectors[:, :dimensions]
        captured = singular[:dimensions].square().sum()
    else:
        eigenvalues, eigenvectors = torch.linalg.eigh(centered.T @ centered)
        basis = eigenvectors[:, -dimensions:].flip(1)
        captured = eigenvalues[-dimensions:].clamp_min(0).sum()
    assert basis.shape == (values.shape[1], dimensions)
    fraction = float((captured / centered.square().sum().clamp_min(1e-12)).item())
    assert 0 <= fraction <= 1.0001
    return center, basis, fraction


def ridge_fit(z, y, regularization=RIDGE):
    z, y = z.double(), y.double()
    mz, my = z.mean(0), y.mean(0)
    x, target = z - mz, y - my
    gram = x.T @ x / len(x) + regularization * torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
    L = torch.linalg.cholesky(gram)
    B = torch.cholesky_solve(x.T @ target / len(x), L)
    return B, my - mz @ B


def crossfit_noise(z, y, concept_ids, check):
    # The PCA/normalization coordinates are fixed on OUTER fit rows.  Only B,b
    # are refit here; no outer-held row or label enters the noise estimate.
    names = sorted(set(concept_ids), key=lambda name: (hashlib.sha256(f'{SEED}|inner|{name}'.encode()).hexdigest(), name))
    assigned = {name: position % 5 for position, name in enumerate(names)}
    labels = torch.tensor([assigned[name] for name in concept_ids], device=z.device)
    predictions = torch.empty_like(y, dtype=torch.float64)
    coverage = torch.zeros(len(y), device=z.device, dtype=torch.int32)
    fold_counts = []
    for inner in range(5):
        check()
        held = labels == inner
        fit = ~held
        assert held.any() and fit.any()
        B, b = ridge_fit(z[fit], y[fit])
        predictions[held] = z[held].double() @ B + b
        coverage[held] += 1
        fold_counts.append(dict(fold=inner, fit_rows=int(fit.sum()), held_rows=int(held.sum())))
    assert torch.all(coverage == 1)
    residuals = y.double() - predictions
    centered = residuals - residuals.mean(0)
    covariance = centered.T @ centered / (len(centered) - 1)
    R = .5 * covariance + .5 * torch.diag(covariance.diagonal())
    R += 1e-4 * torch.eye(R.shape[0], dtype=R.dtype, device=R.device)
    torch.linalg.cholesky(R)
    return R, dict(inner_folds=fold_counts, OOF_coverage_min=1, OOF_coverage_max=1,
        residual_mean_norm=float(residuals.mean(0).norm()), residual_mse=float(residuals.square().mean()),
        covariance_trace=float(covariance.trace()), R_trace=float(R.trace()),
        R_minimum_eigenvalue=float(torch.linalg.eigvalsh(R).min()),
        covariance_definition='0.5 full covariance + 0.5 diagonal covariance + 1e-4 I',
        coordinates='outer-fit unsupervised coordinates fixed; inner crossfit B,b only',
        full_pipeline_OOF_calibration=False, physiological_uncertainty=False)


def gmm_fit(z, components, iterations, check, event, seed=SEED):
    z = z.double()
    generator = torch.Generator(device='cpu').manual_seed(seed)
    initial = torch.randperm(len(z), generator=generator)[:components].to(z.device)
    means = z[initial].clone()
    variances = z.var(0, unbiased=False).clamp_min(VAR_FLOOR)[None].repeat(components, 1)
    weights = torch.full((components,), 1 / components, dtype=z.dtype, device=z.device)
    for step in range(1, iterations + 1):
        check()
        logp = -.5 * (((z[:, None] - means[None]).square() / variances[None]).sum(2)
                       + variances.log().sum(1)[None] + z.shape[1] * math.log(2 * math.pi))
        log_joint = logp + weights.log()[None]
        responsibilities = (log_joint - torch.logsumexp(log_joint, dim=1, keepdim=True)).exp()
        mass = responsibilities.sum(0)
        # No adaptive component removal or selected EM iteration.  Empty/nearly
        # empty components retain a fixed numerical pseudomass, reported below.
        safe_mass = mass.clamp_min(1e-8)
        updated_means = responsibilities.T @ z / safe_mass[:, None]
        second = responsibilities.T @ z.square() / safe_mass[:, None]
        updated_vars = (second - updated_means.square()).clamp_min(VAR_FLOOR)
        means = torch.where((mass > 1e-8)[:, None], updated_means, means)
        variances = torch.where((mass > 1e-8)[:, None], updated_vars, variances)
        weights = safe_mass / safe_mass.sum()
        event(dict(EM_iteration=step, pre_update_training_NLL=float(-torch.logsumexp(log_joint, dim=1).mean()),
            minimum_component_mass=float(mass.min()), maximum_component_mass=float(mass.max()),
            minimum_variance=float(variances.min()), mixture_entropy=float(-(weights * weights.log()).sum()),
            outer_held_metrics_read=False, selected_iteration=False))
    return weights, means, variances


def posterior(y, B, b, R, weights, means, variances):
    """Exact Gaussian conditioning WITHIN the stated fitted latent model.

    z is a row vector and y=z@B+b.  Returns mixture mean, maximum-weight
    component's conditional mean, weights, and all conditional means.
    """
    y, B, b, R = y.double(), B.double(), b.double(), R.double()
    component_means, log_weights = [], []
    q = y.shape[1]
    log2pi = q * math.log(2 * math.pi)
    for k in range(len(weights)):
        # Cov(z,y)=diag(var)@B; Cov(y)=B.T@diag(var)@B+R.
        zy = variances[k, :, None] * B
        evidence_cov = B.T @ zy + R
        L = torch.linalg.cholesky(evidence_cov)
        residual = y - (means[k] @ B + b)
        solved = torch.cholesky_solve(residual.T, L).T
        component_means.append(means[k][None] + solved @ zy.T)
        log_weights.append(weights[k].log() - .5 * (log2pi + 2 * L.diagonal().log().sum()
                                                     + (residual * solved).sum(1)))
    conditional = torch.stack(component_means, dim=1)
    responsibility = torch.stack(log_weights, dim=1).softmax(dim=1)
    averaged = (conditional * responsibility[:, :, None]).sum(1)
    chosen = responsibility.argmax(1)
    component = conditional[torch.arange(len(y), device=y.device), chosen]
    assert torch.isfinite(averaged).all() and torch.isfinite(component).all()
    return averaged, component, responsibility, conditional


def metrics(predictions, targets, items):
    predictions, targets = predictions.float(), targets.float()
    correct, margin = within_concept(predictions, targets, items)
    return dict(mse=(predictions - targets).square().mean(1),
                cosine=F.cosine_similarity(predictions, targets, dim=1),
                same_concept_top1=correct.float(), same_concept_margin=margin)


def summarize(values):
    return {key: float(value.float().mean()) for key, value in values.items()}


def derangement(items):
    positions = {}
    for i, item in enumerate(items):
        positions.setdefault(item['concept_id'], []).append(i)
    result = list(range(len(items)))
    for concept, ids in positions.items():
        assert len(ids) == 10
        offset = 1 + int(hashlib.sha256(f'{SEED}|held_same_concept_shuffle|{concept}'.encode()).hexdigest()[:8], 16) % 9
        for j, index in enumerate(ids):
            result[index] = ids[(j + offset) % len(ids)]
    assert all(i != j for i, j in enumerate(result)) and sorted(result) == list(range(len(items)))
    return result


def self_test():
    torch.manual_seed(SEED)
    d, q, n = 3, 2, 11
    B = torch.randn(d, q, dtype=torch.float64)
    b = torch.randn(q, dtype=torch.float64)
    mean = torch.randn(1, d, dtype=torch.float64)
    variance = torch.rand(1, d, dtype=torch.float64) + .2
    R = torch.eye(q, dtype=torch.float64) * .3
    y = torch.randn(n, q, dtype=torch.float64)
    a, c, w, all_means = posterior(y, B, b, R, torch.ones(1, dtype=torch.float64), mean, variance)
    precision = torch.diag(1 / variance[0]) + B @ torch.linalg.solve(R, B.T)
    rhs = mean / variance + (y - b) @ torch.linalg.solve(R, B.T)
    expected = torch.linalg.solve(precision, rhs.T).T
    assert torch.allclose(a, expected, atol=1e-10) and torch.equal(a, c) and torch.equal(w, torch.ones_like(w))
    weights = torch.tensor([.2, .8], dtype=torch.float64)
    mus = torch.tensor([[-2., 0., 0.], [2., 0., 0.]], dtype=torch.float64)
    vars_ = torch.ones(2, d, dtype=torch.float64)
    zero_B = torch.zeros(d, q, dtype=torch.float64)
    a, c, w, _ = posterior(y, zero_B, b, R, weights, mus, vars_)
    assert torch.allclose(w, weights.expand(n, -1)) and torch.allclose(a, (weights[:, None] * mus).sum(0).expand(n, -1))
    assert torch.allclose(c, mus[1].expand(n, -1))
    zs = torch.randn(60, 3, dtype=torch.float64)
    ys = zs @ B + b
    fitted_B, fitted_b = ridge_fit(zs, ys, 1e-10)
    assert torch.allclose(fitted_B, B, atol=1e-8) and torch.allclose(fitted_b, b, atol=1e-8)
    Rfit, detail = crossfit_noise(zs, ys, [f'concept{i // 10}' for i in range(60)], lambda: None)
    assert torch.linalg.eigvalsh(Rfit).min() > 0 and detail['OOF_coverage_min'] == 1
    events = []
    prior = gmm_fit(zs, 2, 3, lambda: None, events.append)
    assert len(events) == 3 and torch.isclose(prior[0].sum(), torch.tensor(1., dtype=torch.float64))
    assert prior[2].min() >= VAR_FLOOR
    items = [dict(concept_id=f'c{i//10}') for i in range(30)]
    perm = derangement(items)
    assert all(items[i]['concept_id'] == items[j]['concept_id'] for i, j in enumerate(perm))
    # These checks need no CUDA context, input files, or fit result selection.
    print(json.dumps(dict(status='MEASUREMENT_MIXTURE_SYNTHETIC_ALGEBRA_PASSED', checks=8,
                         CUDA_initialized=torch.cuda.is_initialized(), real_data_read=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('trials', 'targets', 'index', 'output'):
        parser.add_argument('--' + key, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-job-name')
    parser.add_argument('--max-seconds', type=float, default=600.)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    assert all((args.trials, args.targets, args.index, args.output, args.gpu_job_name))
    assert args.device == 'cuda:0' and 0 < args.max_seconds <= 600
    assert not args.output.exists(), 'Preserve earlier attempts; use a distinct output root'
    args.output.mkdir(parents=True)
    invoked = time.monotonic()
    stopped = [False]
    signal.signal(signal.SIGTERM, lambda number, frame: stopped.__setitem__(0, True))
    paths = dict(trials=args.trials.resolve(), targets=args.targets.resolve(), index=args.index.resolve(),
                 source=Path(__file__).resolve(), helper_source=ROOT / 'run_BrainHIVE_trialset_fastscreen_20261027_v1.py')
    provenance = {key: dict(path=str(path), sha256=sha(path)) for key, path in paths.items()}
    import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
    D.torch = torch
    auth = ROOT / 'configs/resource_authorization_20261020_v3.json'
    budget_document = dict(gpu_job_name=args.gpu_job_name, maximum_seconds=args.max_seconds,
        maximum_extra_CPU_bytes=10 << 30, maximum_GPU_bytes=20 << 30,
        minimum_CPU_headroom_bytes=10 << 30, maximum_output_bytes=512 << 20,
        resource_authorization=dict(path=str(auth), sha256=sha(auth)))
    wrapper = D.verify_wrapper(budget_document)
    budget = D.Budget(budget_document)
    def check():
        assert not stopped[0] and time.monotonic() - invoked < args.max_seconds, 'Finite model-screen limit reached'
        budget.check()
    def log(event):
        event = event | dict(seconds=time.monotonic() - invoked)
        with (args.output / 'training.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event, allow_nan=False) + '\n')
            stream.flush()
        print(json.dumps(event), flush=True)
    try:
        torch.set_num_threads(2)
        torch.manual_seed(SEED)
        torch.cuda.set_device(torch.device('cuda:0'))
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        source_x = np.load(args.trials, mmap_mode='r', allow_pickle=False)
        source_y = np.load(args.targets, mmap_mode='r', allow_pickle=False)
        assert source_x.dtype == source_y.dtype == np.float32
        items = read_index(args.index)
        contract = validate_index(items, source_x.shape, source_y.shape, 13230)
        assert contract['concepts'] == 1323
        means = []
        for start in range(0, len(items), 256):
            check()
            chunk = np.array(source_x[[row['row'] for row in items[start:start + 256]]], copy=True).mean(1)
            means.append(torch.from_numpy(chunk.reshape(len(chunk), -1)).cuda())
        x = torch.cat(means)
        del means
        targets = torch.from_numpy(np.array(source_y[[row['target_row'] for row in items]], copy=True)).cuda()
        del source_x, source_y
        assert torch.isfinite(x).all() and torch.isfinite(targets).all()
        provenance.update(schema=SCHEMA, seed=SEED, contract=contract, wrapper=wrapper,
            model=dict(EEG_PCA_dim=64, F_PCA_dim=64, GMM_components=16, GMM_diagonal=True,
                EM_steps=30, variance_floor=.05, ridge_lambda=.1, ridge_objective='mean squared error + lambda*||B||^2',
                R_shrinkage=.5, R_jitter=1e-4, inner_concept_folds=5),
            EEG_coordinates='outer-fit physical-R4 mean, outer-fit feature mean/std, randomized PCA64 q72 niter4',
            F_coordinates='outer-fit centered exact covariance PCA64; inverse with outer-fit F mean',
            noise_model='B,b concept-crossfit residual covariance conditional on fixed outer-fit unsupervised coordinates',
            full_pipeline_OOF_uncertainty_calibration=False, physiological_uncertainty=False,
            prediction_coordinates=COORDINATES, inference_uses_target_labels=False,
            outer_held_used_for_selection=False, final_iteration_only=True,
            component_mean_definition='conditional mean of maximum posterior-weight component; NOT exact mixture MAP')
        atomic(args.output / 'provenance.json', provenance)
        outputs = {arm: np.empty((13230, 1024), np.float32) for arm in ARMS}
        constant_predictions = np.empty((13230, 1024), np.float32)
        weights_all = np.empty((13230, 16), np.float64)
        coverage = np.zeros(13230, np.int32)
        fold_results = []
        per_query = []
        for fold in (0, 1):
            check()
            fit_positions = [i for i, row in enumerate(items) if row['fold'] != fold]
            held_positions = [i for i, row in enumerate(items) if row['fold'] == fold]
            fit = torch.tensor(fit_positions, device=x.device)
            held = torch.tensor(held_positions, device=x.device)
            fit_items = [items[i] for i in fit_positions]
            held_items = [items[i] for i in held_positions]
            center_x = x[fit].mean(0)
            scale_x = x[fit].std(0, unbiased=False).clamp_min(1e-4)
            standardized_fit = (x[fit] - center_x) / scale_x
            pca_x_center, pca_x, x_fraction = pca_fit(standardized_fit, DIM, randomized=True, seed=SEED)
            y = ((standardized_fit - pca_x_center) @ pca_x).double()
            y_held = ((((x[held] - center_x) / scale_x) - pca_x_center) @ pca_x).double()
            center_f, pca_f, f_fraction = pca_fit(targets[fit], DIM)
            z = ((targets[fit] - center_f) @ pca_f).double()
            B, b = ridge_fit(z, y)
            R, residual_receipt = crossfit_noise(z, y, [r['concept_id'] for r in fit_items], check)
            prior = gmm_fit(z, COMPONENTS, EM_STEPS, check,
                            lambda event: log(dict(stage='GMM_FIXED_EM', outer_fold=fold, **event)), seed=SEED)
            posterior_average, component_average, responsibilities, _ = posterior(y_held, B, b, R, *prior)
            decoded = {ARMS[0]: posterior_average.float() @ pca_f.T + center_f,
                       ARMS[1]: component_average.float() @ pca_f.T + center_f}
            baseline = center_f[None].expand(len(held), -1)
            constant_metrics = metrics(baseline, targets[held], held_items)
            constant_predictions[held_positions] = baseline.cpu().numpy()
            weights_all[held_positions] = responsibilities.cpu().numpy()
            coverage[held_positions] += 1
            perm = torch.tensor(derangement(held_items), device=x.device)
            entropy = -(responsibilities * responsibilities.clamp_min(1e-300).log()).sum(1)
            distance = (decoded[ARMS[0]] - decoded[ARMS[1]]).norm(dim=1)
            fold_summary = dict(fold=fold, train_rows=len(fit), held_rows=len(held),
                EEG_PCA_explained_variance=x_fraction, F_PCA_explained_variance=f_fraction,
                F_train_mean_squared_norm=float(center_f.square().sum()),
                F_train_mean_total_squared_norm=float(targets[fit].square().sum(1).mean()),
                F_train_centered_covariance_trace=float((targets[fit] - center_f).square().sum() / (len(fit) - 1)),
                R_crossfit=residual_receipt, constant_F_baseline=summarize(constant_metrics), arms={},
                posterior_component_entropy_mean=float(entropy.mean()), maximum_component_weight_mean=float(responsibilities.max(1).values.mean()),
                mean_vs_component_F_distance_mean=float(distance.mean()), selected_EM_iteration=EM_STEPS)
            arm_metrics = {}
            arm_pairing = {}
            for arm in ARMS:
                prediction = decoded[arm]
                value = metrics(prediction, targets[held], held_items)
                centered_cos = F.cosine_similarity(prediction - center_f, targets[held] - center_f, dim=1)
                shuffled_cos = F.cosine_similarity(prediction[perm], targets[held], dim=1)
                gap = value['cosine'] - shuffled_cos
                value['centered_F_cosine'] = centered_cos
                value['correct_minus_same_concept_shuffled_cosine'] = gap
                outputs[arm][held_positions] = prediction.cpu().numpy()
                fold_summary['arms'][arm] = summarize(value)
                arm_metrics[arm] = {key: tensor.cpu().numpy() for key, tensor in value.items()}
                arm_pairing[arm] = gap.cpu().numpy()
            entropy_cpu = entropy.cpu().numpy()
            distance_cpu = distance.cpu().numpy()
            responsibility_cpu = responsibilities.cpu().numpy()
            permutation_cpu = perm.cpu().tolist()
            constant_cpu = {key: value.cpu().numpy() for key, value in constant_metrics.items()}
            for j, original_position in enumerate(held_positions):
                row = items[original_position]
                per_query.append(row | dict(
                    posterior_entropy=float(entropy_cpu[j]), maximum_component_weight=float(responsibility_cpu[j].max()),
                    selected_component=int(responsibility_cpu[j].argmax()), mean_vs_component_F_distance=float(distance_cpu[j]),
                    component_weights=responsibility_cpu[j].tolist(),
                    same_concept_shuffle_row=held_items[permutation_cpu[j]]['row'],
                    constant_F_mse=float(constant_cpu['mse'][j]), constant_F_cosine=float(constant_cpu['cosine'][j]),
                    arms={arm: {key: float(values[j]) for key, values in arm_metrics[arm].items()} for arm in ARMS}))
            fold_results.append(fold_summary)
            checkpoint = dict(schema=SCHEMA, outer_fold=fold, fit_source_rows=[r.get('source_row', r['row']) for r in fit_items],
                held_source_rows=[r.get('source_row', r['row']) for r in held_items],
                EEG_center=center_x.cpu(), EEG_scale=scale_x.cpu(), EEG_PCA_center=pca_x_center.cpu(), EEG_PCA_basis=pca_x.cpu(),
                F_center=center_f.cpu(), F_PCA_basis=pca_f.cpu(), B=B.cpu(), b=b.cpu(), R=R.cpu(),
                GMM_weights=prior[0].cpu(), GMM_means=prior[1].cpu(), GMM_variances=prior[2].cpu(),
                fixed_final_EM_steps=EM_STEPS, outer_held_used_for_fit_or_selection=False)
            torch.save(checkpoint, args.output / f'fold{fold}_model.pt')
            atomic(args.output / f'fold{fold}_summary.json', fold_summary)
            log(dict(stage='OUTER_FOLD_FIXED_FINAL_COMPLETE', **fold_summary))
        assert np.all(coverage == 1)
        index_out = args.output / 'prediction_index.jsonl'
        with index_out.open('x', encoding='utf-8') as stream:
            stream.write(''.join(json.dumps(row) + '\n' for row in items))
        per_query.sort(key=lambda row: row['row'])
        with gzip.open(args.output / 'per_query.jsonl.gz', 'wt', encoding='utf-8') as stream:
            stream.write(''.join(json.dumps(row, allow_nan=False) + '\n' for row in per_query))
        save_npy(args.output / 'posterior_component_weights.npy', weights_all)
        save_npy(args.output / 'constant_F_predictions.npy', constant_predictions)
        summary = dict(status='COMPLETE_EEG_CONDITIONAL_MODEL_SCREEN', schema=SCHEMA, rows_per_arm=13230,
            subject_id=contract['subject_id'], folds=fold_results, arms={}, prediction_index=str(index_out.resolve()),
            prediction_index_sha256=sha(index_out), per_query=str((args.output / 'per_query.jsonl.gz').resolve()),
            per_query_sha256=sha(args.output / 'per_query.jsonl.gz'),
            posterior_weights=dict(path=str((args.output / 'posterior_component_weights.npy').resolve()),
                sha256=sha(args.output / 'posterior_component_weights.npy'), shape=[13230, 16]),
            fixed_final_EM_steps=30, selection_used=False, prediction_coordinates=COORDINATES,
            full_pipeline_OOF_uncertainty_calibration=False, physiological_uncertainty=False,
            common_F_component_dominance_assumed=False, component_mean_is_exact_MAP=False,
            claim_limit='single-subject full-T concept-OOF representation screen; no generated image, SOTA, mechanism or calibrated physical uncertainty claim')
        full_truth = targets.cpu()
        for arm in ARMS:
            raw = outputs[arm]
            assert np.isfinite(raw).all()
            folder = args.output / arm
            folder.mkdir()
            path = folder / 'predictions.npy'
            save_npy(path, raw)
            metric_values = metrics(torch.from_numpy(raw), full_truth, items)
            summary['arms'][arm] = summarize(metric_values) | dict(predictions_path=str(path.resolve()), predictions_sha256=sha(path),
                predictions_shape=[13230, 1024], predictions_norm_mean=float(np.linalg.norm(raw, axis=1).mean()),
                prediction_coordinates=COORDINATES, oof_coverage_min=1, oof_coverage_max=1,
                centered_F_cosine=float(np.mean([row['arms'][arm]['centered_F_cosine'] for row in per_query])),
                correct_minus_same_concept_shuffled_cosine=float(np.mean([row['arms'][arm]['correct_minus_same_concept_shuffled_cosine'] for row in per_query])))
        summary['constant_F_baseline'] = summarize(metrics(torch.from_numpy(constant_predictions), full_truth, items))
        for key, path in paths.items():
            assert sha(path) == provenance[key]['sha256'], 'Input/source changed during fit'
        check()
        assert sum(path.stat().st_size for path in args.output.rglob('*') if path.is_file()) < 512 << 20
        summary.update(seconds=time.monotonic() - invoked, peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,
                       peak_GPU_bytes=budget.peak_GPU_bytes)
        atomic(args.output / 'summary.json', summary)
        atomic(args.output / 'completion.json', dict(status=summary['status'], schema=SCHEMA,
            summary=dict(path=str((args.output / 'summary.json').resolve()), sha256=sha(args.output / 'summary.json')),
            source_sha256=sha(__file__), expected_rows=13230, observed_rows=13230))
        print(json.dumps(summary), flush=True)
        return 0
    except BaseException as error:
        atomic(args.output / 'failure.json', dict(status='FAILED_FIXED_DENOMINATOR_NO_RESULT_SELECTION',
            error=repr(error), traceback=traceback.format_exc(), expected_rows=13230, seconds=time.monotonic() - invoked))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
