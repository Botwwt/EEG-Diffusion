"""D08: repeat-derived EEG reliability modes and closed-form full-F decoding.

No generator backward, new visual encoder, or held-concept fitted statistics.
The fixed output LayerNorm is part of this decoder, not query-target scaling.
"""
import argparse
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
import fit_BrainHIVE_raw_neural_recoverability_20261027_v1 as U
from fit_BrainHIVE_shared_temporal_eeg_20261027_v1 import fold_positions, paired_readout, full_gallery_readout

ROOT = Path(__file__).resolve().parent
SCHEMA = 'BrainHIVE_repeat_reliable_decoder_20261027_v1'
STATUS = 'COMPLETE_REPEAT_RELIABLE_FULL_F_DECODER'
COORDINATES = 'raw1024 full-F ridge decoder with fixed unlearned LayerNorm eps1e-6; no unit-L2 postprocessing or query-target scaling'
sha, ref, atomic = U.sha, U.ref, U.atomic


def temporal_pool(values):
    assert values.shape[-2:] == (17, 250)
    return values.reshape(*values.shape[:-2], 17, 50, 5).mean(-1).flatten(-2)


def reliable_fit(trials, targets, rank=128):
    """All inputs in this function are OUTER-FIT only; four physical repeats."""
    assert trials.ndim == 3 and trials.shape[1] == 4 and targets.shape[0] == len(trials)
    x = trials.double()
    mean4 = x.mean(1)
    center = mean4.mean(0)
    scale = mean4.std(0, unbiased=False).clamp_min(1e-4)
    z = (x - center) / scale
    average = z.mean(1)
    residual = z - average[:, None]
    noise = torch.einsum('nri,nrj->ij', residual, residual) / (len(z) * 3)
    total = average.T @ average / len(z)
    signal_covariance = total - noise / 4
    eye = torch.eye(noise.shape[0], device=noise.device, dtype=noise.dtype)
    noise_regularized = .9 * noise + .1 * noise.trace() / len(noise) * eye
    whitener = U.inverse_sqrt(noise_regularized)
    operator = whitener @ signal_covariance @ whitener
    eigenvalues, vectors = torch.linalg.eigh((operator + operator.T) / 2)
    selected = eigenvalues[-rank:].flip(0)
    assert selected.min() > 0, 'Insufficient positive reliability modes; retain failure, do not change rank'
    projection = whitener @ vectors[:, -rank:].flip(1)
    features = average @ projection
    feature_center = features.mean(0)
    feature_scale = features.std(0, unbiased=False).clamp_min(1e-6)
    features = (features - feature_center) / feature_scale
    target_center = targets.double().mean(0)
    gram = features.T @ features / len(features) + .1 * torch.eye(rank, device=x.device, dtype=x.dtype)
    rhs = features.T @ (targets.double() - target_center) / len(features)
    weight = torch.cholesky_solve(rhs, torch.linalg.cholesky(gram))
    residual_error = float((gram @ weight - rhs).norm() / rhs.norm().clamp_min(1e-12))
    assert residual_error < 1e-8
    return dict(center=center, scale=scale, projection=projection, feature_center=feature_center,
                feature_scale=feature_scale, weight=weight, target_center=target_center,
                selected_signal_to_repeat_noise_eigenvalues=selected,
                positive_reliability_modes=int((eigenvalues > 0).sum()), ridge_solve_relative_error=residual_error)


def predict(mean4, model):
    features = ((mean4.double() - model['center']) / model['scale']) @ model['projection']
    features = (features - model['feature_center']) / model['feature_scale']
    raw = features @ model['weight'] + model['target_center']
    # This architectural output normalization is fixed before fitting and does
    # not use any held target or per-query expected norm. It can still hurt.
    output = F.layer_norm(raw.float(), (1024,), eps=1e-6)
    return output, raw.float()


def self_test():
    torch.set_num_threads(2)
    torch.manual_seed(1729)
    signal_value = torch.randn(240, 6)
    loading = torch.randn(6, 20)
    trials = (signal_value @ loading)[:, None] + .2 * torch.randn(240, 4, 20)
    targets = torch.randn(240, 1024)
    fitted = reliable_fit(trials, targets, rank=6)
    reversed_fit = reliable_fit(trials[:, [3, 1, 0, 2]], targets, rank=6)
    first, _ = predict(trials.mean(1), fitted)
    second, _ = predict(trials.mean(1), reversed_fit)
    assert torch.allclose(first, second, atol=1e-5, rtol=1e-5)
    assert first.shape == (240, 1024) and torch.isfinite(first).all()
    assert torch.allclose(first.mean(1), torch.zeros(240), atol=1e-5)
    assert torch.allclose(first.norm(dim=1), torch.full((240,), 32.), atol=.003)
    mean_only, _ = predict(trials.mean(1), fitted)
    assert torch.equal(first, mean_only), 'Inference consumes the mean4, not the individual trials'
    changed = trials.clone(); changed[200:] += 999
    fit_a = reliable_fit(trials[:200], targets[:200], rank=6)
    fit_b = reliable_fit(changed[:200], targets[:200], rank=6)
    for key in fit_a:
        if isinstance(fit_a[key], torch.Tensor): assert torch.equal(fit_a[key], fit_b[key])
    fixture = torch.arange(17 * 250, dtype=torch.float32).reshape(1, 17, 250)
    pooled = temporal_pool(fixture)
    assert pooled.shape == (1, 850) and float(pooled[0, 0]) == 2 and float(pooled[0, 50]) == 252
    items = [dict(fold=i // 10, concept_id=str(i // 10)) for i in range(20)]
    train, held = fold_positions(items, 0)
    assert train == list(range(10, 20)) and held == list(range(10))
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(status='D08_FIVE_CRITICAL_CPU_CHECKS_PASSED', checks=5, CUDA_initialized=False,
                         checks_scope=['repeat_permutation_invariance', 'mean4_only_inference',
                                       'held_values_do_not_fit_statistics', 'fixed_LayerNorm_coordinates',
                                       'temporal_pool_and_concept_fold_contract'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('trials', 'targets', 'index', 'spec', 'native-parent-binding', 'output'):
        parser.add_argument('--' + key, type=Path)
    parser.add_argument('--gpu-job-name')
    parser.add_argument('--max-seconds', type=float, default=90)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test: self_test(); return 0
    assert all((args.trials, args.targets, args.index, args.spec, args.native_parent_binding, args.output, args.gpu_job_name))
    assert 0 < args.max_seconds <= 90 and not args.output.exists()
    args.output.mkdir(parents=True); started = time.monotonic(); stopped = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))
    import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
    D.torch = torch
    document = dict(gpu_job_name=args.gpu_job_name, maximum_seconds=args.max_seconds,
        maximum_output_bytes=128 << 20, maximum_extra_CPU_bytes=10 << 30,
        maximum_GPU_bytes=20 << 30, minimum_CPU_headroom_bytes=10 << 30,
        resource_authorization=ref(ROOT / 'configs/resource_authorization_20261020_v3.json'))
    wrapper = D.verify_wrapper(document); budget = D.Budget(document)
    def check():
        assert not stopped[0] and time.monotonic() - started < args.max_seconds, 'Finite90s cap; no rank/data reduction on timeout'
        budget.check()
    def log(value):
        value = value | dict(seconds=time.monotonic() - started)
        with (args.output / 'progress.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n'); stream.flush(); os.fsync(stream.fileno())
        print(json.dumps(value, allow_nan=False), flush=True)
    try:
        spec = json.loads(args.spec.read_bytes()); assert spec['schema'] == SCHEMA
        assert spec['reliability_rank'] == 128 and spec['noise_covariance_shrinkage'] == .1 and spec['ridge'] == .1
        for key in ('trials', 'targets', 'index'):
            check(); assert sha(getattr(args, key)) == spec[key + '_sha256']
        items = U.read_index(args.index)
        assert len(items) == 13230 and [row['row'] for row in items] == list(range(13230))
        assert all(row['subject_id'] == 'sub-01' and row['partition'] == 'T' and row['row'] == row['target_row'] for row in items)
        native = json.loads(args.native_parent_binding.read_bytes())
        assert native['prediction_index_sha256'] == sha(args.index) and native['oof_coverage_min'] == native['oof_coverage_max'] == 1
        assert sha(native['predictions_path']) == native['predictions_sha256']
        controls = {name: ref(getattr(args, name)) for name in ('trials', 'targets', 'index', 'spec', 'native_parent_binding')}
        controls['source'] = ref(__file__); controls['wrapper'] = wrapper
        atomic(args.output / 'provenance.json', dict(schema=SCHEMA, controls=controls,
            fits_only_outer_training_concepts=True, held_metrics_select_parameters=False,
            repeat_differences_are_noise_proxy_not_physiological_ground_truth=True,
            normal_inference_target_access=False, inference_uses_four_repeat_mean_only=True,
            fixed_LayerNorm_is_architecture_not_oracle_scaling=True, visual_target_PCA=False,
            original_BrainMLP_reused=False, no_generator_backward=True))
        torch.set_num_threads(2); torch.manual_seed(1729); torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = False; torch.use_deterministic_algorithms(True)
        targets_array = np.load(args.targets, mmap_mode='r', allow_pickle=False)
        trial_array = np.load(args.trials, mmap_mode='r', allow_pickle=False)
        assert trial_array.shape == (13230, 4, 17, 250) and trial_array.dtype == np.float32
        assert targets_array.shape == (13230, 1024) and targets_array.dtype == np.float32
        targets = torch.from_numpy(np.array(targets_array, copy=True)).cuda()
        native_predictions = torch.from_numpy(np.array(np.load(native['predictions_path'], allow_pickle=False), copy=True)).cuda()
        predictions = np.empty((13230, 1024), np.float32); coverage = np.zeros(13230, np.int8); folds = []
        assert torch.isfinite(targets).all() and torch.isfinite(native_predictions).all()
        for fold in (0, 1):
            check(); fit_rows, held_rows = fold_positions(items, fold)
            # Only outer-FIT individual repeats are copied. Held inference only
            # gets the physical mean4, with no held noise covariance estimation.
            raw_fit = np.array(trial_array[fit_rows], copy=True)
            assert np.isfinite(raw_fit).all()
            fit_trials = temporal_pool(torch.from_numpy(raw_fit).cuda()); del raw_fit
            check(); fitted = reliable_fit(fit_trials, targets[fit_rows])
            log(dict(stage='RELIABILITY_AND_FULL_F_RIDGE_FITTED', outer_fold=fold,
                fit_queries=len(fit_rows), selected_rank=128, positive_modes=fitted['positive_reliability_modes'],
                selected_reliability_max=float(fitted['selected_signal_to_repeat_noise_eigenvalues'].max()),
                selected_reliability_min=float(fitted['selected_signal_to_repeat_noise_eigenvalues'].min()),
                ridge_solve_relative_error=fitted['ridge_solve_relative_error']))
            fit_prediction, raw_prediction = predict(fit_trials.mean(1), fitted)
            fit_readout = paired_readout(fit_prediction, targets[fit_rows], [items[i] for i in fit_rows])
            raw_held = np.asarray(trial_array[held_rows]).mean(1)
            assert np.isfinite(raw_held).all()
            held_mean = temporal_pool(torch.from_numpy(np.array(raw_held, copy=True)).cuda())
            held_prediction, held_raw = predict(held_mean, fitted)
            held_readout = paired_readout(held_prediction, targets[held_rows], [items[i] for i in held_rows])
            held_readout.update(full_gallery_readout(held_prediction, torch.tensor(held_rows, device='cuda'), targets, check))
            native_readout = paired_readout(native_predictions[held_rows], targets[held_rows], [items[i] for i in held_rows])
            predictions[held_rows] = held_prediction.cpu().numpy(); coverage[held_rows] += 1
            checkpoint = args.output / f'fold{fold}_decoder.pt'
            torch.save(dict(schema=SCHEMA, model=U.cpu_value(fitted), temporal_pool='mean five consecutive samples ->17x50',
                output_LayerNorm=dict(normalized_shape=1024, eps=1e-6, learned_affine=False),
                fit_source_rows=[items[i]['source_row'] for i in fit_rows], held_source_rows=[items[i]['source_row'] for i in held_rows],
                spec_sha256=sha(args.spec)), checkpoint)
            record = dict(outer_fold=fold, fit_queries=len(fit_rows), held_queries=len(held_rows), fit=fit_readout,
                held=held_readout, matched_native_held=native_readout,
                pre_LayerNorm_held_norm_mean=float(held_raw.norm(dim=1).mean()), checkpoint=ref(checkpoint))
            folds.append(record); log(dict(stage='REPEAT_RELIABLE_FOLD_COMPLETE', **record))
            del fit_trials, fitted, fit_prediction, raw_prediction, held_mean, held_prediction, held_raw
            torch.cuda.empty_cache()
        assert np.all(coverage == 1) and np.isfinite(predictions).all()
        directory = args.output / 'repeat_reliable'; directory.mkdir()
        path = directory / 'predictions.npy'; np.save(path, predictions, allow_pickle=False)
        for key in ('trials', 'targets', 'index', 'spec', 'native_parent_binding'):
            check(); assert sha(getattr(args, key)) == controls[key]['sha256']
        assert sha(native['predictions_path']) == native['predictions_sha256']
        assert sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file()) < 128 << 20
        summary = dict(status=STATUS, schema=SCHEMA, rows=13230, concepts=1323, subject='sub-01', outer_folds=folds,
            prediction_index=str(args.index.resolve()), prediction_index_sha256=sha(args.index),
            arms={'repeat_reliable': dict(predictions_path=str(path.resolve()), predictions_sha256=sha(path),
                predictions_shape=[13230, 1024], prediction_coordinates=COORDINATES,
                oof_coverage_min=1, oof_coverage_max=1)},
            normal_inference_target_access=False, additional_subject_EEG_at_inference=False,
            held_metrics_used_for_selection=False, hyperparameter_sweep=False, fixed_LayerNorm=True,
            no_unit_L2_prediction_postprocessing=True, no_query_target_scaling=True, full1024_visual_targets=True,
            seconds=time.monotonic() - started, peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,
            peak_GPU_bytes=budget.peak_GPU_bytes, provenance=ref(args.output / 'provenance.json'),
            development_only=True, independent_confirmation=False, SOTA_established=False)
        atomic(args.output / 'summary.json', summary); atomic(args.output / 'completion.json', dict(status=STATUS, summary=ref(args.output / 'summary.json')))
        log(dict(stage='COMPLETE', status=STATUS)); return 0
    except BaseException as error:
        atomic(args.output / 'failure.json', dict(status='INCOMPLETE_REPEAT_RELIABLE_DECODER_NO_AUTOMATIC_RETRY',
            error=repr(error), traceback=traceback.format_exc(), seconds=time.monotonic() - started))
        raise


if __name__ == '__main__': raise SystemExit(main())
