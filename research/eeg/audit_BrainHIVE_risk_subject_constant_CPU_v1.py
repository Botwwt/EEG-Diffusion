"""Posthoc Dfit-subject-constant audit of already saved angular-error vectors.

No head replay, calibration, visual-target tensors, EEG/condition caches, model
weights or V scores are read. Only existing D angular-error vectors are used.
A known aggregate-moment preview predates this audit: this is explicitly
not a preregistered result or a new candidate.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time


SCHEMA = 'BrainHIVE_risk_subject_constant_CPU_v1'
STATUS = 'POSTHOC_AUDIT_FIXED_METHOD_NO_MODEL_REVISION'
SUBJECTS = [f'sub-{index:02d}' for index in range(1, 11)]
COUNTS = {'Dfit': 820, 'Dcheck': 830}
LIMITS = dict(max_seconds=120, max_output_bytes=1048576, threads=1)
SOURCE_NAMES = {Path(__file__).name, 'check_BrainHIVE_risk_subject_constant_CPU_v1.py'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def mean(values):
    assert values
    return math.fsum(values) / len(values)


def mse(values, constant):
    return mean([(value - constant) ** 2 for value in values])


def subject_constant_statistics(fit, check, head_aggregate_MSE):
    """Only Dfit means define predictions; Dcheck never defines a predictor."""
    assert set(fit) == set(check) and fit
    for group in (fit, check):
        for values in group.values():
            assert values and all(math.isfinite(value) and 0 <= value <= 2 for value in values)
    assert math.isfinite(head_aggregate_MSE) and 0 <= head_aggregate_MSE <= 4
    pooled_constant = mean([value for subject in sorted(fit) for value in fit[subject]])
    rows, subject_SSE, pooled_SSE, total = [], 0., 0., 0
    for subject in sorted(fit):
        constant = mean(fit[subject])
        subject_MSE = mse(check[subject], constant)
        pooled_MSE = mse(check[subject], pooled_constant)
        count = len(check[subject])
        rows.append(dict(subject=subject, Dfit_queries=len(fit[subject]), Dcheck_queries=count,
            Dfit_subject_constant_risk=constant, Dcheck_observed_mean_risk=mean(check[subject]),
            Dfit_risk_population_std=math.sqrt(mse(fit[subject], constant)),
            Dcheck_risk_population_std=math.sqrt(mse(check[subject], mean(check[subject]))),
            Dcheck_MSE_using_Dfit_subject_constant=subject_MSE,
            Dcheck_MSE_using_Dfit_pooled_constant=pooled_MSE))
        subject_SSE += subject_MSE * count
        pooled_SSE += pooled_MSE * count
        total += count
    subject_MSE, pooled_MSE = subject_SSE / total, pooled_SSE / total
    return dict(per_subject=rows, aggregate=dict(Dfit_queries=sum(map(len, fit.values())),
        Dcheck_queries=total, reconstructed_Dfit_pooled_constant_risk=pooled_constant,
        Dcheck_MSE_using_Dfit_subject_constant=subject_MSE,
        reconstructed_Dcheck_MSE_using_Dfit_pooled_constant=pooled_MSE,
        saved_Dcheck_calibrated_head_MSE=head_aggregate_MSE,
        subject_constant_minus_pooled_constant_MSE=subject_MSE - pooled_MSE,
        head_minus_subject_constant_MSE=head_aggregate_MSE - subject_MSE,
        head_minus_pooled_constant_MSE=head_aggregate_MSE - pooled_MSE,
        subject_constant_relative_MSE_reduction_vs_pooled=(pooled_MSE - subject_MSE) / pooled_MSE if pooled_MSE else None,
        head_relative_MSE_reduction_vs_subject_constant=(subject_MSE - head_aggregate_MSE) / subject_MSE if subject_MSE else None,
        head_relative_MSE_reduction_vs_pooled_constant=(pooled_MSE - head_aggregate_MSE) / pooled_MSE if pooled_MSE else None))


def validate_registration(registration):
    assert registration['schema'] == SCHEMA + '_registration'
    assert registration['status'] == STATUS
    assert registration['subjects'] == SUBJECTS
    assert registration['limits'] == LIMITS
    assert set(registration['source_sha256']) == SOURCE_NAMES
    assert isinstance(registration['known_aggregate_moment_preview'], dict)
    inputs = registration['geometry_inputs']
    assert len(inputs) == 20
    assert {(row['subject'], row['split']) for row in inputs} == {
        (subject, split) for subject in SUBJECTS for split in COUNTS}
    assert len({str(Path(row['path']).resolve()) for row in inputs}) == 20
    for row in inputs:
        assert row['count'] == COUNTS[row['split']]
        assert Path(row['path']).is_absolute()
        assert len(row['array_rows_sha256']) == 64
    for row in inputs + [registration['calibration_input'], registration['geometry_summary_input']]:
        assert Path(row['path']).is_absolute()
        assert len(row['sha256']) == 64 and all(c in '0123456789abcdef' for c in row['sha256'])
    assert Path(registration['output_path']).is_absolute()
    assert Path(registration['output_path']).suffix == '.json'
    assert not Path(registration['output_path']).exists(), 'Preserve existing audit; no overwrite'


def run(registration_path):
    started = time.monotonic()
    def time_check():
        assert time.monotonic() - started < LIMITS['max_seconds'], 'Fixed 120-second CPU audit budget exhausted'
    registration_path = Path(registration_path).resolve()
    registration_sha = sha(registration_path)
    registration = json.loads(registration_path.read_text(encoding='utf-8'))
    validate_registration(registration)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    pins = {str(registration_path): registration_sha}
    for name, expected in registration['source_sha256'].items():
        path = Path(__file__).parent / name
        assert sha(path) == expected, 'Audit source drift: ' + name
        pins[str(path.resolve())] = expected
    entries = registration['geometry_inputs'] + [registration['calibration_input'], registration['geometry_summary_input']]
    for row in entries:
        time_check()
        assert sha(row['path']) == row['sha256'], 'Input drift: ' + row['path']
        pins[row['path']] = row['sha256']
    calibration = json.loads(Path(registration['calibration_input']['path']).read_text())
    geometry = json.loads(Path(registration['geometry_summary_input']['path']).read_text())
    assert calibration['kind'] == 'Dfit_nonnegative_affine_prediction_risk_calibration'
    assert calibration['risk_is_physiological_SNR'] is False
    assert calibration['Dcheck_used_to_revise_or_select_candidate'] is False
    assert calibration['Dcheck_queries'] == 8300
    assert geometry['status'] == 'ALL10_FULL_ALLOWED_T_OOF_AND_D_NATIVE_GEOMETRY_AUDITED_CPU'
    moments = {}
    for row in geometry['per_subject']:
        if row['split'] in COUNTS:
            key = (row['subject'], row['split'])
            assert key not in moments
            moments[key] = row
    assert set(moments) == {(subject, split) for subject in SUBJECTS for split in COUNTS}
    # Only the 20 registered geometry .pt files are loaded. No models are imported.
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    time_check()
    vectors = {'Dfit': {}, 'Dcheck': {}}
    split_ids = {}
    moment_checks = []
    for row in sorted(registration['geometry_inputs'], key=lambda item: (item['subject'], item['split'])):
        time_check()
        payload = torch.load(row['path'], weights_only=True, map_location='cpu')
        subject, split = row['subject'], row['split']
        assert payload['subject'] == subject and payload['split'] == split
        ids_tensor = payload['array_rows']
        assert isinstance(ids_tensor, torch.Tensor) and ids_tensor.device.type == 'cpu' and ids_tensor.ndim == 1
        assert ids_tensor.dtype in (torch.int32, torch.int64)
        ids = ids_tensor.tolist()
        assert len(ids) == row['count'] == COUNTS[split]
        assert len(set(ids)) == len(ids) and all(isinstance(i, int) and i >= 0 for i in ids)
        ids_sha = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
        assert ids_sha == row['array_rows_sha256']
        if split not in split_ids:
            split_ids[split] = set(ids)
        assert set(ids) == split_ids[split], 'Each subject must have the same split row-ID set'
        cosine = payload['values']['token_cosine']
        assert isinstance(cosine, torch.Tensor) and cosine.device.type == 'cpu'
        assert cosine.dtype == torch.float32 and cosine.shape == (COUNTS[split],)
        assert torch.isfinite(cosine).all() and not cosine.requires_grad
        risk = 1 - cosine  # Reproduce the historical FP32 y, then use float64 arithmetic for audit sums.
        assert torch.isfinite(risk).all() and torch.all((risk >= 0) & (risk <= 2))
        values = risk.double().tolist()
        vectors[split][subject] = values
        moment = moments[(subject, split)]
        assert moment['queries'] == COUNTS[split]
        old = moment['statistics']['token_cosine']
        observed_mean = float(cosine.double().mean())
        observed_sample_std = float(cosine.double().std(unbiased=True))
        assert abs(observed_mean - old['mean']) <= 2e-6
        assert abs(observed_sample_std - old['std']) <= 2e-6
        moment_checks.append(dict(subject=subject, split=split, array_rows_sha256=ids_sha,
            token_cosine_mean_difference=observed_mean - old['mean'],
            token_cosine_sample_std_difference=observed_sample_std - old['std']))
        del payload, ids_tensor, cosine, risk
    assert split_ids['Dfit'].isdisjoint(split_ids['Dcheck'])
    result = subject_constant_statistics(vectors['Dfit'], vectors['Dcheck'], calibration['Dcheck_calibrated_risk_MSE'])
    aggregate = result['aggregate']
    constant_difference = aggregate['reconstructed_Dfit_pooled_constant_risk'] - calibration['Dfit_constant_risk']
    pooled_MSE_difference = aggregate['reconstructed_Dcheck_MSE_using_Dfit_pooled_constant'] - calibration['Dcheck_Dfit_constant_MSE']
    assert abs(constant_difference) <= 2e-7
    assert abs(pooled_MSE_difference) <= 2e-7
    for split in COUNTS:
        assert geometry['aggregate'][split]['queries'] == COUNTS[split] * 10
        reconstructed_cosine_mean = mean([1 - value for subject in SUBJECTS for value in vectors[split][subject]])
        assert abs(reconstructed_cosine_mean - geometry['aggregate'][split]['token_cosine']) <= 2e-6
    for path, expected in pins.items():
        time_check()
        assert sha(path) == expected, 'Input/source changed during audit: ' + path
    time_check()
    output = dict(schema=SCHEMA, status=STATUS, complete=True,
        registration_path=str(registration_path), registration_sha256=registration_sha,
        known_aggregate_moment_preview=registration['known_aggregate_moment_preview'],
        preview_known_before_registration=True, pre_result_registration=False,
        per_subject=result['per_subject'], aggregate=aggregate,
        saved_calibration_comparators=dict(Dfit_constant_risk=calibration['Dfit_constant_risk'],
            Dcheck_Dfit_constant_MSE=calibration['Dcheck_Dfit_constant_MSE'],
            Dcheck_calibrated_risk_MSE=calibration['Dcheck_calibrated_risk_MSE']),
        checks=dict(all10_Dfit820_Dcheck830_loaded=True, all_split_row_ID_sets_match=True,
            Dfit_Dcheck_row_ID_sets_disjoint=True, array_rows_SHA_verified=True,
            calibration_Dfit_constant_difference=constant_difference,
            calibration_Dcheck_pooled_MSE_difference=pooled_MSE_difference,
            calibration_absolute_tolerance=2e-7, geometry_moment_absolute_tolerance=2e-6,
            geometry_moment_checks=moment_checks, all_source_and_input_SHAs_rechecked_after_reads=True),
        immutable_input_and_source_sha256=pins,
        execution=dict(CPU_only=True, CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],
            torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
            torch_version=torch.__version__, seconds=time.monotonic() - started, limits=LIMITS,
            torch_load_calls=20, weights_only=True, map_location='cpu'),
        read_policy=dict(registered_geometry_tensors=20, registered_calibration_JSON=True,
            registered_geometry_summary_JSON=True, registration_and_source_hash_reads=True,
            head_replayed=False, EEG_or_condition_caches_read=False, target_images_read=False,
            model_weights_read=False, V_scores_read=False, calibration_or_training_performed=False),
        interpretation=dict(
            method='Predict each Dcheck y=1-token_cosine using its subject mean y calculated only on Dfit.',
            head='Use the previously saved aggregate Dcheck calibrated-head MSE exactly; no new head outputs or affine fit.',
            relative_improvements='Arithmetic relative MSE reductions, not causal attribution or explained causal fractions.',
            subject_control='This controls only a constant subject mean; it does not identify all subject effects or prove within-subject predictive correlation.',
            paired_confidence_interval_available=False, within_subject_head_correlation_available=False,
            reason='Per-query head predictions are not read or available in this audit.',
            picture_generation_gain_established=False, physiological_SNR_established=False,
            model_or_gate_revision=False, new_candidate=False, final_test_selection=False,
            no_image_or_SOTA_conclusion=True))
    data = (json.dumps(output, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    assert len(data) <= LIMITS['max_output_bytes']
    time_check()
    output_path = Path(registration['output_path'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('xb') as stream:
        stream.write(data)
    print(json.dumps(dict(status=STATUS, output_path=str(output_path), output_sha256=sha(output_path), aggregate=aggregate)), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', required=True)
    args = parser.parse_args()
    run(args.registration)


if __name__ == '__main__':
    main()
