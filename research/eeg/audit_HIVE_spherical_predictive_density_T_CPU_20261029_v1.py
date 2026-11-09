"""Full true-OOF-T directional-density falsification; zero CUDA/new EEG fits."""
import argparse
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--protocol', type=Path, required=True)
args = parser.parse_args()
protocol = json.loads(args.protocol.read_bytes())
ROOT = Path(protocol['source_root'])
sys.path.insert(0, protocol['host_source_root'])
sys.path.insert(0, str(ROOT))
import brainhive_paired_study_contract_20261025_v1 as C
C.setup_runtime(dict(runtime_manifest=dict(path=protocol['runtime_manifest'], sha256=protocol['required_sha256'][protocol['runtime_manifest']])))
import numpy as np
import torch
torch.set_num_threads(2)
from brainhive_spherical_predictive_density_CPU_20261029_v1 import (
    fit_vmf_concentration, host_direction, restore_host,
    vmf_cosine_quantiles, vmf_log_normalizer,
    fit_marginal_vmf_tilt, vmf_tilt_score,
)

start = time.monotonic()
def deadline(signum, frame):
    raise TimeoutError('Fixed CPU audit deadline reached')
signal.signal(signal.SIGALRM, deadline)
signal.alarm(protocol['max_seconds'])
out = Path(protocol['output_root'])
assert out.is_dir() and not (out / 'completion.json').exists()

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def write(name, value):
    target = out / name
    pending = target.with_name(target.name + '.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    pending.replace(target)

def guard():
    assert time.monotonic() - start < protocol['max_seconds'] - 5
    assert not torch.cuda.is_initialized()
    memory = dict((line.split(':')[0], int(line.split()[1]) * 1024) for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split()) >= 2 and line.split()[1].isdigit())
    assert memory['MemAvailable'] > 4 * 1024 ** 3

def bootstrap(values):
    assert values.shape == (10, 1323, 10)
    averages = values.mean(axis=2)
    rng = np.random.default_rng(1729)
    estimates = np.array([averages[np.ix_(rng.integers(10, size=10), rng.integers(1323, size=1323))].mean() for unused in range(1000)])
    return dict(mean=float(values.mean()), descriptive_subject_concept_95=np.quantile(estimates, [.025, .975]).tolist(), per_subject=averages.mean(axis=1).tolist())

def quantile_lookup(dimension, concentrations):
    low, high = float(concentrations.min()), float(concentrations.max())
    if high - low < 1e-9:
        q = vmf_cosine_quantiles(dimension, low)
        return np.tile(q, (len(concentrations), 1)), 0.0
    knots = np.linspace(low, high, 33)
    table = np.array([vmf_cosine_quantiles(dimension, k) for k in knots])
    error = 0.
    for k in np.linspace(low, high, 9)[1:-1] + (high-low)/512:
        exact = vmf_cosine_quantiles(dimension, k)
        predicted = np.array([np.interp(k, knots, table[:, j]) for j in range(3)])
        error = max(error, float(np.max(abs(exact - predicted))))
    assert error < 1e-5, ('fixed_quantile_interpolation_guard', error)
    return np.stack([np.interp(concentrations, knots, table[:, j]) for j in range(3)], axis=1), error

try:
    for path, digest in protocol['required_sha256'].items():
        assert sha(path) == digest, path
    assert len(protocol['sources_sha256']) == 3
    for name, digest in protocol['sources_sha256'].items():
        assert sha(ROOT / name) == digest
    contract = json.loads(Path(protocol['input_contract']).read_bytes())
    split = json.loads(Path(protocol['split_manifest']).read_bytes())
    assert contract['status'] == 'ALL10_COMPLETE_REAL_OOF_AND_SEPARATE_D_INPUTS_BOUND_NO_FIT'
    array_rows = split['image_rows']['T']
    assert len(array_rows) == 13230 and len(contract['subjects']) == 10
    position = {row: index for index, row in enumerate(array_rows)}
    entries = [json.loads(line) for line in Path(protocol['T_index']).read_text().splitlines()]
    assert len(entries) == 13230 and {row['array_row'] for row in entries} == set(array_rows)
    by_row = {row['array_row']: row for row in entries}
    assert all(by_row[row]['partition'] == 'T' for row in array_rows)
    concepts = [by_row[row]['concept_id'] for row in array_rows]
    counts = collections.Counter(concepts)
    assert len(counts) == 1323 and set(counts.values()) == {10}
    concept_names = sorted(counts)
    density_fold = {concept: int(hashlib.sha256(('HIVE_directional_density_v1|' + str(concept)).encode()).hexdigest()[:8], 16) % 2 for concept in concept_names}
    target = torch.load(contract['target_cache_path'], map_location='cpu', mmap=True, weights_only=True)
    assert tuple(target.shape) == (16540, 1024) and sha(contract['target_cache_path']) == contract['target_cache_sha256']
    raw_tilt = protocol.get('native_contrastive_marginal_tilt', False)
    dimension = 1024 if raw_tilt else 1023
    columns = 10 if raw_tilt else 8
    metrics = np.full((10, 13230, columns), np.nan, dtype=np.float64)
    coverage = np.zeros((10, 13230), dtype=np.int8)
    source_reports = []
    for subject_index, subject in enumerate(contract['subjects']):
        role_count = 0
        for source in subject['sources']:
            if source['role'] not in ('oof0', 'oof1'):
                continue
            guard()
            role_count += 1
            assert sha(source['condition_cache_path']) == source['condition_cache_sha256']
            assert sha(source['checkpoint_path']) == source['checkpoint_sha256']
            cache = torch.load(source['condition_cache_path'], map_location='cpu', mmap=True, weights_only=True)
            checkpoint = torch.load(source['checkpoint_path'], map_location='cpu', mmap=True, weights_only=True)
            assert cache['model_sha256'] == source['checkpoint_sha256']
            weight = checkpoint['model']['block.layers.0.norm.weight'].numpy().astype(np.float64)
            bias = checkpoint['model']['block.layers.0.norm.bias'].numpy().astype(np.float64)
            assert np.min(abs(weight)) > 1e-6
            records = source['groups']['component_train']
            rows = np.array([record['array_row'] for record in records])
            cache_rows = np.array([record['cache_row'] for record in records])
            destination = np.array([position[int(row)] for row in rows])
            assert np.array_equal(cache['array_rows'][cache_rows].numpy(), rows)
            condition = cache['condition'][cache_rows].numpy().astype(np.float64)
            visual = target[rows].numpy().astype(np.float64)
            mean, radius, offset = host_direction(condition, weight, bias)
            visual_direction, visual_radius, visual_offset = host_direction(visual, weight, bias)
            assert np.max(abs(restore_host(mean, radius, weight, bias) - condition)) < 1e-5
            roundtrip_error = float(np.max(abs(restore_host(mean, radius, weight, bias) - condition)))
            projection_error = float(np.mean(np.linalg.norm(restore_host(visual_direction, radius, weight, bias) - visual, axis=1) / np.linalg.norm(visual, axis=1)))
            if raw_tilt:
                mean = condition / np.linalg.norm(condition, axis=1, keepdims=True)
                visual_direction = visual / np.linalg.norm(visual, axis=1, keepdims=True)
            cosine = np.einsum('ij,ij->i', mean, visual_direction)
            groups = collections.defaultdict(list)
            for index, row in enumerate(rows):
                groups[by_row[int(row)]['concept_id']].append(index)
            assert all(len(indices) == 10 for indices in groups.values())
            donor = np.arange(len(rows))
            for indices in groups.values():
                ordered = sorted(indices, key=lambda index: int(rows[index]))
                donor[ordered] = np.roll(ordered, -1)
            donor_cosine = np.einsum('ij,ij->i', mean[donor], visual_direction)
            folds = np.array([density_fold[by_row[int(row)]['concept_id']] for row in rows])
            report = dict(subject=subject['subject'], role=source['role'], query_count=len(rows), condition_SHA256=source['condition_cache_sha256'], checkpoint_SHA256=source['checkpoint_sha256'], maximum_condition_roundtrip_abs=roundtrip_error, maximum_center_offset=float(np.max(abs(offset))), mean_visual_projection_relative_error=projection_error, density_uses_native_raw_normalized_InfoNCE_coordinates=raw_tilt, folds=[])
            for heldout in (0, 1):
                fit = folds != heldout
                check = folds == heldout
                assert fit.sum() > 100 and check.sum() > 100
                conditional_concentration = fit_vmf_concentration(dimension, float(cosine[fit].mean()))
                population_vector = visual_direction[fit].mean(axis=0)
                resultant = np.linalg.norm(population_vector)
                assert 1e-12 < resultant < 1
                population_direction = population_vector / resultant
                population_concentration = fit_vmf_concentration(dimension, float(resultant))
                population_cosine = visual_direction[check] @ population_direction
                direct_score = vmf_log_normalizer(dimension, conditional_concentration) + conditional_concentration * cosine[check]
                population_score = vmf_log_normalizer(dimension, population_concentration) + population_concentration * population_cosine
                extra = {}
                if raw_tilt:
                    EEG_population = mean @ population_direction
                    tilt = fit_marginal_vmf_tilt(dimension, population_concentration, EEG_population[fit], cosine[fit])
                    conditional_score, lengths, axis_cosine = vmf_tilt_score(dimension, population_concentration, tilt, EEG_population[check], cosine[check], population_cosine)
                    donor_score, unused_lengths, unused_axis = vmf_tilt_score(dimension, population_concentration, tilt, EEG_population[donor][check], donor_cosine[check], population_cosine)
                    quantile_rows, quantile_error = quantile_lookup(dimension, lengths)
                    inside = ((axis_cosine >= quantile_rows[:, 0]) & (axis_cosine <= quantile_rows[:, 2])).astype(float)
                    quantiles = np.stack((quantile_rows.min(axis=0), quantile_rows.max(axis=0)))
                    values = np.stack((conditional_score, population_score, conditional_score-population_score, conditional_score-donor_score, inside, axis_cosine, donor_cosine[check], lengths, cosine[check], direct_score),axis=1)
                    extra = dict(EEG_tilt_scale=tilt, posterior_kappa_range=[float(lengths.min()),float(lengths.max())], maximum_quantile_interpolation_error=quantile_error, direct_EEG_axis_minus_population_mean_nats=float((direct_score-population_score).mean()))
                else:
                    conditional_score = direct_score
                    quantiles = vmf_cosine_quantiles(dimension, conditional_concentration)
                    inside = ((cosine[check] >= quantiles[0]) & (cosine[check] <= quantiles[2])).astype(float)
                    values = np.stack((conditional_score, population_score, conditional_score - population_score, conditional_concentration * (cosine[check] - donor_cosine[check]), inside, cosine[check], donor_cosine[check], np.full(check.sum(), conditional_concentration)), axis=1)
                metrics[subject_index, destination[check]] = values
                coverage[subject_index, destination[check]] += 1
                report['folds'].append(dict(heldout_density_fold=heldout, fit_queries=int(fit.sum()), check_queries=int(check.sum()), conditional_kappa=conditional_concentration, population_kappa=population_concentration, predictive_cosine_quantiles=quantiles.tolist(), empirical_95_coverage=float(inside.mean()), conditional_minus_population_mean_nats=float((conditional_score - population_score).mean()), **extra))
            source_reports.append(report)
            del condition, visual, mean, visual_direction, cache, checkpoint
            write('progress.json', dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), source_count=len(source_reports), expected_sources=20, seconds=time.monotonic()-start, CUDA_initialized=torch.cuda.is_initialized()))
        assert role_count == 2
    assert np.all(coverage == 1) and np.isfinite(metrics).all()
    # A rectangular identity-preserving layout for independent reanalysis.
    order = np.array([position[row] for concept in concept_names for row in sorted(array_rows, key=int) if by_row[row]['concept_id'] == concept])
    assert len(order) == 13230 and len(set(order)) == 13230
    rectangle = metrics[:, order].reshape(10, 1323, 10, columns)
    summaries = {name: bootstrap(rectangle[..., index]) for name, index in [('conditional_minus_population_logscore_nats', 2), ('correct_minus_sameconcept_donor_logscore_nats', 3), ('predictive_95_coverage', 4)]}
    support = summaries['conditional_minus_population_logscore_nats']['descriptive_subject_concept_95'][0] > 0 and summaries['correct_minus_sameconcept_donor_logscore_nats']['descriptive_subject_concept_95'][0] > 0 and summaries['predictive_95_coverage']['descriptive_subject_concept_95'][0] <= .95 <= summaries['predictive_95_coverage']['descriptive_subject_concept_95'][1]
    column_names = ['conditional_logscore', 'population_logscore', 'conditional_minus_population', 'correct_minus_donor', 'inside_nominal95', 'posterior_axis_cosine' if raw_tilt else 'correct_cosine', 'donor_cosine', 'posterior_kappa' if raw_tilt else 'conditional_kappa']
    if raw_tilt:
        column_names += ['native_EEG_target_cosine', 'direct_EEG_axis_logscore']
    np.savez_compressed(out / 'all132300_T_density_predictions.npz', values=rectangle, array_rows=np.array(array_rows)[order], concept_ids=np.array([str(name) for name in concept_names]), columns=np.array(column_names))
    write('all20_sources_and_density_folds.json', source_reports)
    result = dict(status='COMPLETE_FULL10_TRUE_OOF_T_NATIVE_MARGINAL_TILT_DENSITY' if raw_tilt else 'COMPLETE_FULL10_TRUE_OOF_T_DIRECTIONAL_DENSITY_FALSIFICATION', UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), queries=132300, subjects=10, T_stimuli=13230, concepts=1323, role_models=20, ambient_dimension=dimension, summaries=summaries, simple_isotropic_conditional_density_supported=bool(support), nested_conditional_logscore_support=bool(summaries['conditional_minus_population_logscore_nats']['descriptive_subject_concept_95'][0]>0 and summaries['correct_minus_sameconcept_donor_logscore_nats']['descriptive_subject_concept_95'][0]>0), action='ALLOW_SEPARATE_NONLINEAR_SEMANTIC_PROFILE_DESIGN' if support else 'STOP_USING_THIS_ISOTROPIC_DENSITY_AS_CALIBRATED_EEG_PRIOR', all_query_coverage_exactly_once=True, heldout_density_concepts_excluded_from_density_calibration=True, upstream_condition_is_real_OOF=True, visual_F_is_projected_to_real_own_LN_sphere=not raw_tilt, native_contrastive_raw_normalized_coordinates=raw_tilt, no_Dfit_Dcheck_V_official_result_read=True, new_EEG_models=0, new_EEG_updates=0, new_images=0, new_GPU_calls=0, CUDA_initialized=torch.cuda.is_initialized(), NEW_contacted=False, novelty_established=False, seconds=time.monotonic()-start, protocol_SHA256=sha(args.protocol), outputs=[dict(name=name, bytes=(out/name).stat().st_size, SHA256=sha(out/name)) for name in ('all132300_T_density_predictions.npz', 'all20_sources_and_density_folds.json')])
    guard()
    write('completion.json', result)
    signal.alarm(0)
    print(json.dumps(result))
except BaseException as error:
    import traceback
    write('failure.json', dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc(), seconds=time.monotonic()-start, CUDA_initialized=torch.cuda.is_initialized()))
    raise
