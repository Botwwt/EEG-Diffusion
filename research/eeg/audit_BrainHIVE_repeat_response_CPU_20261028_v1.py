"""Independent CPU audit of the fixed T-only EEG-repeat response diagnostic.

The saved 256 Q/K/V traces are recomputed with NumPy float64 softmax attention.
All 44,800 metric rows are checked for complete identity coverage and aggregated
without selecting layers, times or concepts.  The remaining metric rows do not
contain Q/K/V: their coverage and arithmetic, not their forward computation, are
independently audited.  This is not an image-effect or iid-noise verification.
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np
import torch

import brainhive_gain_calibration_20261027_v1 as U


ARMS = ('native', 'ordinary')
RESPONSE_NAMES = ('base', 'minus', 'plus')
FIELDS = tuple(name + '_' + field for name in RESPONSE_NAMES for field in
               ('positive_MSE', 'J', 'within_concept_top1', 'norm')) + (
                   'correction_relative_norm', 'quadratic_scaling_residual')
IDENTITY_FIELDS = ('query_index', 'concept_id', 'stimulus_id', 'arm', 'step', 'layer')
TRACE_LAYERS = (0, 23, 46, 69)
ATOL = 1e-4
RTOL = 1e-4


def numpy_attention(q, k, v):
    """Independent stable softmax(q k^T / sqrt(d)) v, with no torch math."""
    assert q.ndim == k.ndim == v.ndim == 4
    assert q.shape[0] == 1 and k.shape[0] == v.shape[0] == 21
    assert k.shape == v.shape and q.shape[1] == k.shape[1]
    assert q.shape[2] == 16 and k.shape[2] == 4
    assert q.shape[-1] == k.shape[-1]
    logits = np.matmul(q, np.swapaxes(k, -1, -2)) / np.sqrt(q.shape[-1])
    weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    response = np.matmul(weights, v)
    assert response.shape == (21, q.shape[1], 16, q.shape[-1])
    assert np.isfinite(response).all()
    return response


def numpy_metrics(response, correct):
    """Re-derive the fixed response metrics from float64 responses."""
    assert response.shape[0] == 21 and 0 <= correct < 10
    base = response[0]
    delta3 = response[1:5].mean(axis=0) - base
    delta2 = response[5:11].mean(axis=0) - base
    teachers = response[11:21]
    target = teachers[correct]
    wrong = np.delete(teachers, correct, axis=0)
    denominator = max(float(np.square(wrong - target).mean()), 1e-10)
    answer = {}
    for name, value in (('base', base), ('minus', base - 3 * delta3),
                        ('plus', base + 3 * delta3)):
        distances = np.square(teachers - value).reshape(10, -1).mean(axis=1)
        positive = float(distances[correct])
        negative = float(np.delete(distances, correct).mean())
        answer[name + '_positive_MSE'] = positive
        answer[name + '_J'] = (negative - positive) / denominator
        answer[name + '_within_concept_top1'] = int(np.argmin(distances) == correct)
        answer[name + '_norm'] = float(np.linalg.norm(value.ravel()))
    answer['correction_relative_norm'] = float(
        3 * np.linalg.norm(delta3.ravel()) / max(np.linalg.norm(base.ravel()), 1e-10))
    answer['quadratic_scaling_residual'] = float(
        np.linalg.norm((delta2 - 3 * delta3).ravel()) /
        max(np.linalg.norm(delta2.ravel()) + 3 * np.linalg.norm(delta3.ravel()), 1e-10))
    assert set(answer) == set(FIELDS) and all(np.isfinite(v) for v in answer.values())
    return answer


def cpu_array(tensor):
    assert isinstance(tensor, torch.Tensor) and tensor.device.type == 'cpu'
    assert tensor.is_floating_point() and torch.isfinite(tensor).all()
    return tensor.numpy().astype(np.float64, copy=False)


def interval(values, bootstrap_indices):
    values = np.asarray(values, dtype=np.float64)
    assert values.shape == (8,)
    draws = values[bootstrap_indices].mean(axis=1)
    lo, hi = np.quantile(draws, [.025, .975])
    return dict(mean=float(values.mean()), descriptive_95_interval=[float(lo), float(hi)],
                positive_concepts=int((values > 0).sum()), zero_concepts=int((values == 0).sum()),
                negative_concepts=int((values < 0).sum()), all8_concept_values=values.tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--completion', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized() and not args.output.exists()
    completion_ref = U.ref(args.completion)
    done = U.read(U.checked(completion_ref))
    assert done['status'] == 'D26_TONLY_PHYSICAL_REPEAT_SAMEQ_RESPONSE_DIAGNOSTIC_COMPLETE'
    assert done['query_count'] == 80 and done['concept_count'] == 8 and done['subjects'] == 1
    assert done['new_images'] == done['epochs'] == 0 and done['new_UNet_calls'] == 644
    for key in ('Dcheck_or_V_read', 'ordinary_changed', 'sampler_changed',
                'attention_responses_changed', 'image_gain_established',
                'iid_repeat_noise_established', 'formal_D_resumed'):
        assert done[key] is False, key
    assert done['observer_latent_bitwise_parity'] is True
    spec = U.read(U.checked(done['spec']))
    verified_pins = []
    for pin in spec['source_pins']:
        U.checked(pin)
        verified_pins.append(pin)
    freeze = U.read(U.checked(spec['ordinary_freeze']))
    U.checked(freeze['checkpoint'])
    U.checked(freeze['source'])
    assert freeze['selected_strength'] == 1
    refs = [completion_ref, done['spec'], done['metrics'], done['traces'], done['input_bank']]
    for item in refs:
        U.checked(item)
    bank = torch.load(U.checked(done['input_bank']), map_location='cpu', weights_only=False)
    queries = bank['queries']
    corrects = bank['corrects']
    assert len(queries) == len(corrects) == 80
    assert len({q['stimulus_id'] for q in queries}) == 80
    assert len({q['row'] for q in queries}) == 80
    assert all(q['partition'] == 'T' and q['subject_id'] == 'sub-01' for q in queries)
    concepts = list(dict.fromkeys(q['concept_id'] for q in queries))
    assert len(concepts) == 8 and set(Counter(q['concept_id'] for q in queries).values()) == {10}
    for concept in concepts:
        assert sorted(corrects[i] for i, q in enumerate(queries) if q['concept_id'] == concept) == list(range(10))
    for name, shape in [('conditions', (80, 2, 11, 1024)), ('means', (80, 11, 4250)),
                        ('tokens', (80, 2, 11, 4, 2048)),
                        ('teacher_tokens', (80, 10, 4, 2048))]:
        assert tuple(bank[name].shape) == shape, name
        assert bank[name].device.type == 'cpu' and torch.isfinite(bank[name]).all(), name
    # Recover the original four observations algebraically from the full and R3
    # means.  M3 uses lexicographic subsets, so deletion order is 3,2,1,0.
    means = cpu_array(bank['means'])
    recovered_trials = 4 * means[:, :1] - 3 * means[:, 1:5][:, ::-1]
    pair_means = np.stack([recovered_trials[:, pair].mean(axis=1)
                           for pair in combinations(range(4), 2)], axis=1)
    assert np.allclose(means[:, 5:11], pair_means, atol=2e-5, rtol=2e-5)
    assert np.allclose(means[:, 0], recovered_trials.mean(axis=1), atol=2e-5, rtol=2e-5)
    subset_max_abs_error = float(np.max(np.abs(means[:, 5:11] - pair_means)))
    del recovered_trials, pair_means, means

    rows = []
    seen = {}
    cube = np.full((80, 2, 4, 70, len(FIELDS)), np.nan, dtype=np.float64)
    with U.checked(done['metrics']).open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            assert set(row) == set(IDENTITY_FIELDS) | set(FIELDS)
            qi, arm, step, layer = (row[k] for k in ('query_index', 'arm', 'step', 'layer'))
            assert all(type(v) is int for v in (qi, arm, step, layer))
            assert 0 <= qi < 80 and 0 <= arm < 2 and 0 <= step < 4 and 0 <= layer < 70
            key = (qi, arm, step, layer)
            assert key not in seen, key
            assert row['concept_id'] == queries[qi]['concept_id']
            assert row['stimulus_id'] == queries[qi]['stimulus_id']
            values = np.asarray([row[k] for k in FIELDS], dtype=np.float64)
            assert np.isfinite(values).all()
            for name in RESPONSE_NAMES:
                assert row[name + '_within_concept_top1'] in (0, 1)
                assert row[name + '_positive_MSE'] >= 0 and row[name + '_norm'] >= 0
            assert row['correction_relative_norm'] >= 0
            assert 0 <= row['quadratic_scaling_residual'] <= 1 + 1e-6
            cube[qi, arm, step, layer] = values
            seen[key] = row
            rows.append(row)
    assert len(rows) == len(seen) == 44800 and np.isfinite(cube).all()
    assert set(seen) == {(q, a, s, l) for q in range(80) for a in range(2)
                         for s in range(4) for l in range(70)}

    traces = torch.load(U.checked(done['traces']), map_location='cpu', weights_only=False)
    assert isinstance(traces, list) and len(traces) == 256
    expected_trace_keys = {(i, a, s, l) for i in range(80) if corrects[i] == 0
                           for a in range(2) for s in range(4) for l in TRACE_LAYERS}
    assert len(expected_trace_keys) == 256
    trace_seen = set()
    per_trace_errors = []
    largest_absolute = {k: 0. for k in FIELDS}
    largest_relative = {k: 0. for k in FIELDS}
    layer_names = {}
    for trace in traces:
        key = tuple(trace[k] for k in ('query_index', 'arm', 'step', 'layer'))
        assert key in expected_trace_keys and key not in trace_seen
        trace_seen.add(key)
        assert trace['correct'] == corrects[key[0]] == 0
        layer_names.setdefault(key[3], trace['name'])
        assert layer_names[key[3]] == trace['name']
        assert set(trace['metrics']) == set(FIELDS)
        got = numpy_metrics(numpy_attention(cpu_array(trace['q']), cpu_array(trace['k']),
                                            cpu_array(trace['v'])), trace['correct'])
        errors = {}
        for name in FIELDS:
            expected = float(trace['metrics'][name])
            assert expected == float(seen[key][name]), (key, name, 'saved trace/row mismatch')
            absolute = abs(got[name] - expected)
            relative = absolute / max(abs(expected), 1e-10)
            if name.endswith('_within_concept_top1'):
                assert got[name] == expected, (key, name, got[name], expected)
            else:
                assert absolute <= ATOL + RTOL * abs(expected), (key, name, got[name], expected)
            largest_absolute[name] = max(largest_absolute[name], absolute)
            largest_relative[name] = max(largest_relative[name], relative)
            errors[name] = dict(numpy_float64=got[name], saved_torch=expected,
                                absolute_error=absolute, relative_error=relative)
        per_trace_errors.append(dict(query_index=key[0], arm=key[1], step=key[2],
                                     layer=key[3], metrics=errors))
    assert trace_seen == expected_trace_keys

    # Each query has the same complete four-step, 70-layer denominator.  Then
    # average ten stimuli per concept and all eight concepts, with equal weight.
    query_means = cube.mean(axis=(2, 3))
    concept_means = np.stack([query_means[[i for i, q in enumerate(queries)
                                         if q['concept_id'] == concept]].mean(axis=0)
                              for concept in concepts])
    overall = concept_means.mean(axis=0)
    for arm, label in enumerate(ARMS):
        assert set(done['summary'][label]) == set(FIELDS)
        for fi, name in enumerate(FIELDS):
            assert np.isclose(overall[arm, fi], done['summary'][label][name], atol=1e-10, rtol=1e-10), (label, name)
    boot = np.random.default_rng(1729).integers(0, 8, size=(1000, 8))
    aggregation = {}
    for arm, label in enumerate(ARMS):
        absolute = {name: float(overall[arm, fi]) for fi, name in enumerate(FIELDS)}
        contrasts = {}
        for target_name, source_name in [('minus', 'base'), ('plus', 'base'), ('minus', 'plus')]:
            metrics_out = {}
            for metric in ('J', 'positive_MSE', 'within_concept_top1', 'norm'):
                left = FIELDS.index(target_name + '_' + metric)
                right = FIELDS.index(source_name + '_' + metric)
                metrics_out[metric] = interval(concept_means[:, arm, left] - concept_means[:, arm, right], boot)
            contrasts[target_name + '_minus_' + source_name] = metrics_out
        per_concept = [dict(concept_id=concept, stimuli=10,
                            absolute={name: float(concept_means[ci, arm, fi])
                                      for fi, name in enumerate(FIELDS)},
                            minus_minus_base={metric: float(concept_means[ci, arm, FIELDS.index('minus_' + metric)] -
                                                           concept_means[ci, arm, FIELDS.index('base_' + metric)])
                                              for metric in ('J', 'positive_MSE', 'within_concept_top1', 'norm')})
                       for ci, concept in enumerate(concepts)]
        aggregation[label] = dict(absolute=absolute, contrasts=contrasts, concepts=per_concept)
    primary = aggregation['ordinary']['contrasts']['minus_minus_base']
    directional = primary['J']['mean'] > 0 and primary['positive_MSE']['mean'] <= 0
    decision = dict(action=('DIRECTIONAL_SUPPORT_IN_THIS_DIAGNOSTIC' if directional else
                            'STOP_THIS_DIAGNOSTIC'),
                    primary='ordinary minus response versus original ordinary response',
                    locked_rule='equal-concept mean J increment >0 AND positive teacher MSE increment <=0',
                    J_increment=primary['J']['mean'], positive_MSE_increment=primary['positive_MSE']['mean'],
                    bootstrap_intervals_are_descriptive_not_independent_confirmation=True,
                    rule_does_not_require_or_establish_significance=True,
                    automatic_image_queue_authorized=False, image_efficacy_established=False,
                    iid_repeat_noise_established=False, whole_generator_unbiasedness_established=False,
                    novelty_established=False)
    for item in refs:
        U.checked(item)
    U.checked(freeze['checkpoint'])
    U.checked(freeze['source'])
    assert not torch.cuda.is_initialized()
    args.output.mkdir(parents=True)
    U.atomic(args.output / 'trace_numeric_comparisons.json', dict(
        status='ALL256_SAVED_QKV_TRACES_NUMPY_FLOAT64_RECOMPUTED',
        absolute_tolerance=ATOL, relative_tolerance=RTOL,
        acceptance='abs_error <= atol + rtol * abs(saved); Top1 exactly equal',
        maximum_absolute_errors=largest_absolute, maximum_relative_errors=largest_relative,
        comparisons=per_trace_errors))
    U.atomic(args.output / 'per_query_equal_weight_means.json', dict(
        queries=[dict(query_index=i, concept_id=q['concept_id'], stimulus_id=q['stimulus_id'],
                      arms={label: {name: float(query_means[i, ai, fi]) for fi, name in enumerate(FIELDS)}
                            for ai, label in enumerate(ARMS)}) for i, q in enumerate(queries)],
        steps_per_query=4, layers_per_step=70, selection_of_layers_or_steps=False))
    result = dict(status='D26_ALL80_SAMEQ_REPEAT_RESPONSE_NUMERICS_AND_AGGREGATION_CPU_VERIFIED',
                  source=U.ref(__file__), completion=completion_ref, spec=done['spec'],
                  input_files=refs, all_spec_source_pins_verified=verified_pins,
                  ordinary_checkpoint=freeze['checkpoint'], ordinary_source=freeze['source'],
                  ordinary_strength=freeze['selected_strength'], ordinary_changed=False,
                  metric_rows=44800, saved_QKV_traces_independently_recomputed=256,
                  other_response_rows_forward_recomputed=False,
                  complete_identity_coverage=True, duplicate_metric_rows=0,
                  complete_trace_identity_coverage=True, layer_names=layer_names,
                  subset_mean_max_absolute_consistency_error=subset_max_abs_error,
                  NumPy_float64_attention=True, numerical_maximum_absolute_errors=largest_absolute,
                  numerical_maximum_relative_errors=largest_relative,
                  aggregation_order='equal mean four times x70 layers within query; ten queries per concept; eight concepts',
                  aggregation=aggregation, decision=decision,
                  bootstrap=dict(repetitions=1000, seed=1729, implementation='NumPy default_rng PCG64',
                                 unit='eight whole concepts', shared_indices_for_all_contrasts=True,
                                 independent_confirmation=False),
                  trace_comparisons=U.ref(args.output / 'trace_numeric_comparisons.json'),
                  query_means=U.ref(args.output / 'per_query_equal_weight_means.json'),
                  new_images=0, new_training_updates=0, GPU_initialized=False,
                  Dcheck_or_V_effects_read=False, sampler_changed=False,
                  attention_responses_changed=False, formal_D_resumed=False,
                  automatic_image_queue=False, image_gain_established=False,
                  seconds=time.monotonic() - started)
    U.atomic(args.output / 'summary.json', result)
    print(json.dumps(dict(status=result['status'], decision=decision,
                          metric_rows=result['metric_rows'], seconds=result['seconds'])))


if __name__ == '__main__':
    main()
