"""Independent, T-only NumPy64 audit of six frozen OOF condition banks.

The prediction axes are [stimulus, native/ordinary, zero/aligned/wrong, F].
This program never loads a model, imports a GPU framework, fits a parameter,
reads Dcheck/V scores, or admits image generation automatically.  T-only
directional admission is necessary evidence for this frontend, not a novelty
claim or a claim that images improve.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np


METHODS = ('native', 'ordinary')
FRONTENDS = ('zero', 'aligned', 'wrong_assignment')
METRICS = ('raw_F_MSE', 'raw_F_cosine', 'same_concept_Top1', 'same_concept_2AFC')
COMPARISONS = (('aligned_minus_zero', 1, 0),
               ('aligned_minus_wrong', 1, 2),
               ('wrong_minus_zero', 2, 0))
N_STIMULI, N_CONCEPTS, F_DIM = 13230, 1323, 1024
BOOTSTRAPS, BOOTSTRAP_SEED = 1000, 1729
ORDINARY_CHECKPOINT_SHA256 = 'a66869e5676ccea27f326f4f0b86cd446526b42460734905990758e49e11cc03'
ORDINARY_SOURCE_SHA256 = '2b93c0644bde268ae876f0e2b722106852b25910fedc1cffc97820e96cadf5d7'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            hasher.update(part)
    return hasher.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size)


def check_ref(item):
    require(isinstance(item, dict) and 'path' in item and 'sha256' in item,
            'A path/SHA256 source reference is required')
    path = Path(item['path'])
    require(path.is_file(), 'Missing referenced source: ' + str(path))
    require(digest(path) == item['sha256'], 'SHA256 mismatch: ' + str(path))
    if 'bytes' in item:
        require(path.stat().st_size == item['bytes'], 'Byte-count mismatch: ' + str(path))
    return path


def exclusive_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def unit_rows(values):
    values = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    result = np.zeros_like(values)
    np.divide(values, norm, out=result, where=norm > 0.)
    return result


def concept_metrics(predictions, targets):
    """Compute per-stimulus metrics; exact ties receive fractional credit.

    predictions: [10,2,3,1024], targets: [10,1024].  The 2AFC denominator
    is exactly the other nine images in this concept, never all T images.
    Zero-norm cosine is defined as zero and is counted in the output audit.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    require(predictions.shape == (10, 2, 3, F_DIM) and targets.shape == (10, F_DIM),
            'Expected ten complete stimuli for every concept')
    normalized_targets = unit_rows(targets)
    normalized_predictions = unit_rows(predictions)
    scores = np.einsum('imfd,jd->imfj', normalized_predictions,
                       normalized_targets, optimize=True)
    diagonal = scores[np.arange(10), :, :, np.arange(10)]
    maxima = scores.max(axis=-1)
    tie_count = (scores == maxima[..., None]).sum(axis=-1)
    top1 = (diagonal == maxima) / tie_count
    # Subtract the self-comparison (always a tie) from the fractional rank.
    afc = ((diagonal[..., None] > scores).sum(axis=-1)
           + .5 * (diagonal[..., None] == scores).sum(axis=-1) - .5) / 9.
    mse = ((predictions - targets[:, None, None, :]) ** 2).mean(axis=-1)
    values = np.stack((mse, diagonal, top1, afc), axis=-1)
    require(np.isfinite(values).all(), 'Nonfinite NumPy64 metric')
    return values, dict(zero_prediction_norms=int((np.linalg.norm(predictions, axis=-1) == 0.).sum()),
                        zero_target_norms=int((np.linalg.norm(targets, axis=-1) == 0.).sum()),
                        tied_top1_queries=int((tie_count > 1).sum()))


def mathematical_self_check():
    """Analytical perfect/opposite/tie cases, independent of saved experiments."""
    targets = np.eye(10, F_DIM, dtype=np.float64)
    predictions = np.broadcast_to(targets[:, None, None, :], (10, 2, 3, F_DIM)).copy()
    perfect, _ = concept_metrics(predictions, targets)
    require(np.array_equal(perfect[..., 0], np.zeros((10, 2, 3))), 'Perfect MSE identity')
    require(np.array_equal(perfect[..., 1:], np.ones((10, 2, 3, 3))), 'Perfect retrieval identity')
    opposite, _ = concept_metrics(-predictions, targets)
    require(np.array_equal(opposite[..., 2:], np.zeros((10, 2, 3, 2))), 'Opposite retrieval identity')
    tied, _ = concept_metrics(np.zeros_like(predictions), targets)
    require(np.all(tied[..., 2] == .1) and np.all(tied[..., 3] == .5), 'Fractional tie identity')
    return dict(perfect_retrieval=True, opposite_retrieval=True,
                all_ties_Top1=.1, all_ties_2AFC=.5)


def validate_index(path):
    index = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    require(len(index) == N_STIMULI, 'Incomplete T denominator')
    groups = {}
    for position, row in enumerate(index):
        for field in ('row', 'target_row', 'source_row', 'array_row', 'fold'):
            require(isinstance(row.get(field), int) and not isinstance(row[field], bool),
                    'Index integer field missing: ' + field)
        require(row['row'] == row['target_row'] == position, 'Compact/target order mismatch')
        require(row['source_row'] == row['array_row'], 'Original EEG/target source mismatch')
        require(row.get('partition') == 'T' and row.get('subject_id') == 'sub-01',
                'Only registered sub-01 T rows are allowed')
        require(row['fold'] in (0, 1), 'Unknown concept fold')
        require(isinstance(row.get('concept_id'), str) and row['concept_id'], 'Missing concept identity')
        require(isinstance(row.get('stimulus_id'), str) and row['stimulus_id'], 'Missing stimulus identity')
        groups.setdefault(row['concept_id'], []).append(position)
    require(len(groups) == N_CONCEPTS, 'Incomplete T concepts')
    require(len({row['array_row'] for row in index}) == N_STIMULI, 'Duplicate original rows')
    require(len({row['stimulus_id'] for row in index}) == N_STIMULI, 'Duplicate stimulus identity')
    require({row['fold'] for row in index} == {0, 1}, 'Both OOF folds are required')
    for rows in groups.values():
        require(len(rows) == 10, 'Every T concept requires all ten stimuli')
        require(len({index[i]['fold'] for i in rows}) == 1, 'A concept crosses OOF folds')
    return index, groups


def validate_oof_contract(contract, index, completion):
    require(contract.get('status') == 'ALL10_COMPLETE_REAL_OOF_AND_SEPARATE_D_INPUTS_BOUND_NO_FIT',
            'Unexpected native input contract')
    subjects = contract.get('subjects', [])
    require(len(subjects) == 10, 'Native contract must cover all ten original subjects')
    subject = subjects[0]
    require(subject.get('subject') == 'sub-01', 'First native contract subject is not sub-01')
    sources = {row['role']: row for row in subject['sources'] if row['role'] in ('oof0', 'oof1')}
    require(set(sources) == {'oof0', 'oof1'}, 'Two original OOF source roles are required')
    lookup = {row['array_row']: row for row in index}
    coverage, counts, proofs, source_refs = {}, {}, [], []
    for role in ('oof0', 'oof1'):
        source = sources[role]
        role_rows = source['groups']['component_train']
        expected = {row['array_row'] for row in index if row['fold'] == int(role[-1])}
        actual = [row['array_row'] for row in role_rows]
        require(set(actual) == expected and len(actual) == len(expected), 'OOF role/fold coverage mismatch: ' + role)
        require(len({row['cache_row'] for row in role_rows}) == len(role_rows), 'Duplicate OOF cache row: ' + role)
        for row in role_rows:
            require(isinstance(row['cache_row'], int) and row['cache_row'] >= 0, 'Invalid OOF cache position')
            require(lookup[row['array_row']]['fold'] == int(role[-1]), 'Non-held fold routed to OOF encoder')
            require(row['array_row'] not in coverage, 'Duplicate OOF row route')
            coverage[row['array_row']] = role
        for kind in ('checkpoint', 'condition_cache'):
            item = dict(path=source[kind + '_path'], sha256=source[kind + '_sha256'])
            check_ref(item)
            source_refs.append(item)
        counts[role] = len(role_rows)
        proofs.append(dict(role=role, held_fold=int(role[-1]), queries=len(role_rows),
                           held_concepts=len({lookup[i]['concept_id'] for i in actual}),
                           checkpoint_sha256=source['checkpoint_sha256'],
                           condition_cache_sha256=source['condition_cache_sha256']))
    require(set(coverage) == set(lookup), 'OOF route omits a T stimulus')
    require(completion.get('role_query_counts') == counts, 'Completion OOF role counts disagree')
    return counts, proofs, source_refs


def interval(values, lower, upper):
    return dict(mean=np.asarray(values).tolist(), descriptive_95_lower=np.asarray(lower).tolist(),
                descriptive_95_upper=np.asarray(upper).tolist())


def run(completion_path, output):
    start = time.monotonic()
    require(not output.exists(), 'Output already exists; no overwrite or hidden rerun')
    self_check = mathematical_self_check()
    completion_ref = reference(completion_path)
    done = read_json(completion_path)
    require(done.get('new_models') == 0 and done.get('training_updates') == 0
            and done.get('new_images') == 0, 'Only read-only frozen-model condition measurement is allowed')
    require(done.get('ordinary_changed') is False and done.get('formal_D_resumed') is False,
            'Ordinary must stay frozen and formal D must stay stopped')
    require(done.get('all_weights_frozen') is True, 'Explicit frozen-weight proof is required')
    require(isinstance(done.get('max_mean4_cache_error'), (int, float))
            and 0. <= done['max_mean4_cache_error'] <= 3e-5, 'Mean4 cache replay exceeds tolerance')
    checked = {}
    for key in ('predictions', 'index', 'targets', 'CPU_measurement', 'ordinary_freeze', 'spec', 'native_input_contract'):
        checked[key] = check_ref(done[key])
    spec = read_json(checked['spec'])
    require(isinstance(spec.get('source_pins'), list) and len(spec['source_pins']) > 0,
            'Nonempty registered source pins are required')
    for pin in spec['source_pins']:
        check_ref(pin)
    pinned = {(str(Path(pin['path']).resolve()), pin['sha256']) for pin in spec['source_pins']}
    for key in ('index', 'targets', 'native_input_contract'):
        require((str(checked[key].resolve()), done[key]['sha256']) in pinned,
                'Condition measurement input is absent from registered source pins: ' + key)
    if 'axes' in done:
        require(done['axes'] == dict(condition_method=list(METHODS), EEG_frontend=list(FRONTENDS)),
                'Prediction bank axes differ from the fixed six-arm contract')
    frozen = read_json(checked['ordinary_freeze'])
    require(frozen['checkpoint']['sha256'] == ORDINARY_CHECKPOINT_SHA256
            and frozen['source']['sha256'] == ORDINARY_SOURCE_SHA256,
            'The original frozen ordinary component identity changed')
    require(frozen.get('selected_strength') == 1. and frozen.get('radius_rho_before_strength') == .25,
            'Ordinary strength/radius changed')
    require(frozen.get('change_ordinary_loss_or_weights') is False
            and frozen.get('new_training_authorized_for_this_component') is False,
            'Ordinary freeze declaration changed')
    check_ref(frozen['checkpoint'])
    check_ref(frozen['source'])
    index, groups = validate_index(checked['index'])
    counts, oof_proofs, model_refs = validate_oof_contract(read_json(checked['native_input_contract']), index, done)
    predictions = np.load(checked['predictions'], mmap_mode='r', allow_pickle=False)
    targets = np.load(checked['targets'], mmap_mode='r', allow_pickle=False)
    require(predictions.shape == (N_STIMULI, 2, 3, F_DIM) and predictions.dtype == np.float32,
            'Expected float32 [13230,2,3,1024] prediction bank')
    require(targets.shape == (N_STIMULI, F_DIM) and targets.dtype == np.float32,
            'Expected float32 [13230,1024] T targets')
    per_query = np.empty((N_STIMULI, 2, 3, len(METRICS)), dtype=np.float64)
    concept_values = np.empty((N_CONCEPTS, 2, 3, len(METRICS)), dtype=np.float64)
    prediction_sum = np.zeros((2, 3, F_DIM), dtype=np.float64)
    prediction_square_sum = np.zeros((2, 3), dtype=np.float64)
    numeric = dict(zero_prediction_norms=0, zero_target_norms=0, tied_top1_queries=0)
    concept_rows = []
    # First occurrence order binds every concept to the original compact index.
    for ci, (concept, rows) in enumerate(groups.items()):
        actual = np.asarray(predictions[rows], dtype=np.float64)
        truth = np.asarray(targets[rows], dtype=np.float64)
        require(np.isfinite(actual).all() and np.isfinite(truth).all(), 'Nonfinite conditions or T targets')
        values, notes = concept_metrics(actual, truth)
        for key in numeric:
            numeric[key] += notes[key]
        per_query[rows] = values
        concept_values[ci] = values.mean(axis=0)
        prediction_sum += actual.sum(axis=0)
        prediction_square_sum += (actual * actual).sum(axis=(0, 3))
        concept_rows.append(dict(concept_id=concept, fold=index[rows[0]]['fold'], stimulus_rows=rows,
                                 array_rows=[index[i]['array_row'] for i in rows],
                                 stimulus_ids=[index[i]['stimulus_id'] for i in rows],
                                 metrics=concept_values[ci].tolist()))
    overall = concept_values.mean(axis=0)
    mean_prediction = prediction_sum / N_STIMULI
    average_energy = prediction_square_sum / N_STIMULI
    mean_energy = (mean_prediction * mean_prediction).sum(axis=-1)
    mean_fraction = np.zeros_like(mean_energy)
    np.divide(mean_energy, average_energy, out=mean_fraction, where=average_energy > 0.)
    require(np.all(mean_fraction >= 0.) and np.all(mean_fraction <= 1. + 1e-12), 'Invalid mean energy fraction')
    differences = np.stack([per_query[:, :, a] - per_query[:, :, b] for _, a, b in COMPARISONS], axis=2)
    concept_differences = np.stack([concept_values[:, :, a] - concept_values[:, :, b]
                                    for _, a, b in COMPARISONS], axis=2)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty((BOOTSTRAPS, 2, 3, len(METRICS)), dtype=np.float64)
    paired_draws = np.empty_like(draws)
    for bi in range(BOOTSTRAPS):
        sampled = rng.integers(0, N_CONCEPTS, size=N_CONCEPTS)
        draws[bi] = concept_values[sampled].mean(axis=0)
        paired_draws[bi] = concept_differences[sampled].mean(axis=0)
    low, high = np.quantile(draws, [.025, .975], axis=0)
    paired_low, paired_high = np.quantile(paired_draws, [.025, .975], axis=0)
    paired_mean = concept_differences.mean(axis=0)
    top1 = METRICS.index('same_concept_Top1')
    method_gates = {}
    for mi, method in enumerate(METHODS):
        correct_vs_zero = bool(paired_mean[mi, 0, top1] > 0. and paired_low[mi, 0, top1] > 0.)
        correct_vs_wrong = bool(paired_mean[mi, 1, top1] > 0. and paired_low[mi, 1, top1] > 0.)
        mse_not_worse = bool(overall[mi, 1, 0] <= overall[mi, 0, 0])
        method_gates[method] = dict(aligned_minus_zero_Top1_positive_with_positive_lower95=correct_vs_zero,
                                    aligned_minus_wrong_Top1_positive_with_positive_lower95=correct_vs_wrong,
                                    aligned_raw_F_MSE_mean_not_worse_than_zero=mse_not_worse,
                                    direction_gate=correct_vs_zero and correct_vs_wrong and mse_not_worse)
    primary_pass = method_gates['ordinary']['direction_gate']
    # An offline direction gate never creates a generator admission artifact.
    decision = ('T_ONLY_DIRECTION_GATE_PASSED_NO_AUTOMATIC_IMAGE_ADMISSION' if primary_pass
                else 'NO_IMAGE_ADMISSION_THIS_FRONTEND')
    for pin in spec['source_pins']:
        check_ref(pin)
    for key in checked:
        check_ref(done[key])
    for item in model_refs + [frozen['checkpoint'], frozen['source']]:
        check_ref(item)
    require(reference(completion_path) == completion_ref, 'Completion changed during audit')
    output.mkdir(parents=True)
    with (output / 'all_per_query_and_concepts.npz').open('xb') as stream:
        np.savez_compressed(stream, per_query=per_query, per_query_paired=differences,
                            per_concept=concept_values, per_concept_paired=concept_differences,
                            bootstrap_means=draws, bootstrap_paired=paired_draws,
                            compact_rows=np.arange(N_STIMULI, dtype=np.int64),
                            array_rows=np.array([row['array_row'] for row in index], dtype=np.int64),
                            folds=np.array([row['fold'] for row in index], dtype=np.int8),
                            mean_condition=mean_prediction, condition_average_squared_norm=average_energy,
                            condition_mean_energy_fraction=mean_fraction)
    with (output / 'all1323_concepts.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
        for ci, row in enumerate(concept_rows):
            row['paired_differences'] = concept_differences[ci].tolist()
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    summary = dict(status='ALL13230_T_SIX_FROZEN_OOF_CONDITION_METRICS_CPU_COMPLETE',
                   completed_UTC=datetime.now(timezone.utc).isoformat(),
                   source=reference(__file__), completion=completion_ref, sources=done,
                   axes=dict(condition_method=list(METHODS), EEG_frontend=list(FRONTENDS),
                             metric=list(METRICS), paired_comparison=[r[0] for r in COMPARISONS]),
                   independent_arithmetic='NumPy float64; source arrays float32; no model or GPU framework imported',
                   unique_T_stimuli=N_STIMULI, complete_concepts=N_CONCEPTS, stimuli_per_concept=10,
                   role_query_counts=counts, original_OOF_routes_verified=oof_proofs,
                   raw_F_coordinates='Actual original F coordinates; no centring/affine removal/refit',
                   cosine_zero_norm_convention='zero cosine, included in full denominator',
                   same_concept_gallery=10, top1_ties='exact float64 maximum ties receive 1 / tie_count',
                   AFC_ties='true target beats each of nine alternatives: win 1, exact tie 0.5',
                   numeric=numeric, metrics=interval(overall, low, high),
                   paired=interval(paired_mean, paired_low, paired_high),
                   condition_overall_mean_energy_fraction=mean_fraction.tolist(),
                   condition_mean_energy_fraction_definition='||mean_T(c)||^2 / mean_T(||c||^2); no covariance interpretation',
                   bootstrap=dict(unit='complete concept', equal_concept_weight=True, samples=BOOTSTRAPS,
                                  seed=BOOTSTRAP_SEED, interval='descriptive percentile [0.025,0.975]',
                                  shared_draws_across_all_arms_and_paired_differences=True),
                   primary='ordinary_aligned', method_direction_gates=method_gates,
                   direction_gate=bool(primary_pass), decision=decision,
                   image_admission_written=False, automatic_generation_authorized=False,
                   interpretation='Frozen T-only measurement; passing this direction gate cannot establish novelty, independent generalization, or image benefit.',
                   all_sources_SHA256_reverified_unchanged=True, mathematical_self_check=self_check,
                   new_models=0, training_updates=0, new_images=0, GPU_jobs=0,
                   ordinary_changed=False, formal_D_resumed=False,
                   all_per_query_and_concepts=reference(output / 'all_per_query_and_concepts.npz'),
                   all1323_concepts=reference(output / 'all1323_concepts.jsonl'),
                   seconds=time.monotonic() - start)
    exclusive_json(output / 'summary.json', summary)
    print(json.dumps(dict(status=summary['status'], decision=decision,
                          ordinary=method_gates['ordinary'], seconds=summary['seconds']), allow_nan=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completion', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.completion, args.output)


if __name__ == '__main__':
    main()
