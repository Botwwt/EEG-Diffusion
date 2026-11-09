"""Independent raw replay plus registered even/odd and equal-dimension controls."""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
from collections import defaultdict
import datetime
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ARMS = ('unchanged', 'correct_phase', 'opposite_phase', 'other_repeat_phase', 'reversed_gradient_fit',
        'same_parameter_even_only', 'same_parameter_odd_only', 'one_parameter_even_fit', 'one_parameter_gain_fit')


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def check(item):
    path = Path(item['path'])
    assert path.is_file() and sha(path) == item['sha256']
    if 'bytes' in item:
        assert path.stat().st_size == item['bytes']
    return path


def ref(path):
    return dict(path=str(path), bytes=Path(path).stat().st_size, sha256=sha(path))


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def peak_rss():
    return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:'))) * 1024


def segment(x, offset):
    # Intentionally independent implementation; no primary module imported.
    values = np.asarray(x[:, :, offset - 1:offset + 66], np.float64)
    d = values[:, :, 1:] - values[:, :, :-1]
    return d[:, :, 1:65], d[:, :, :64], d[:, :, 2:66]


def fit(x, y, before, after, wrong=False):
    g = [after - y, y - before]
    if wrong:
        g = [z[:, :, ::-1] for z in g]
    candidate, loss = [], []
    error = x - y
    for gradient, low, high in [(g[0], 0, .5), (g[1], -.5, 0)]:
        a = gradient.reshape(len(x), -1)
        e = error.reshape(len(x), -1)
        numerator, energy = np.einsum('ij,ij->i', a, e), np.einsum('ij,ij->i', a, a)
        delta = np.clip(np.divide(numerator, energy, out=np.zeros(len(x)), where=energy > 1e-30), low, high)
        candidate.append(delta)
        loss.append(np.square(e - delta[:, None] * a).sum(1))
    use0 = (loss[0] < loss[1]) | ((loss[0] == loss[1]) & (np.abs(candidate[0]) <= np.abs(candidate[1])))
    return np.where(use0, candidate[0], candidate[1])


def predict(y, before, after, delta, wrong=False):
    a, b = after - y, y - before
    if wrong:
        a, b = a[:, :, ::-1], b[:, :, ::-1]
    return y + delta[:, None, None] * np.where(delta[:, None, None] >= 0, a, b)


def one_column_fit(column, error, low, high):
    c = column.reshape(len(column), -1)
    e = error.reshape(len(error), -1)
    numerator, denominator = np.einsum('ij,ij->i', c, e), np.einsum('ij,ij->i', c, c)
    return np.clip(np.divide(numerator, denominator, out=np.zeros(len(c)), where=denominator > 1e-30), low, high)


def run(spec_path):
    start = time.monotonic()
    base_rss = peak_rss()
    spec = read(spec_path)
    assert spec['status'] == 'D34_MECHANISM_CONTROL_AND_INDEPENDENT_REPLAY_LOCKED'
    for item in spec['source_pins']:
        check(item)
    primary = read(check(spec['primary_completion']))
    upstream_spec = read(check(spec['primary_spec']))
    assert primary['status'] == 'D34_ALL_T_FIXED_PAIR_SAMPLING_PHASE_EXPLANATION_COMPLETE'
    output = Path(spec['output'])
    assert not output.exists()
    output.mkdir(parents=True)
    raw = np.load(check(upstream_spec['T_trials']), mmap_mode='r', allow_pickle=False).reshape(52920, 17, 250)
    phases = np.load(check(primary['all_phases']), mmap_mode='r', allow_pickle=False)
    original_errors = np.load(check(primary['all_held_errors']), mmap_mode='r', allow_pickle=False)
    d33 = read(check(upstream_spec['upstream_D33_completion']))
    index = [json.loads(x) for x in check(upstream_spec['T_index']).read_text().splitlines()]
    groups = defaultdict(list)
    for i, row in enumerate(index):
        assert row['row'] == i and row['partition'] == 'T' and row['subject_id'] == 'sub-01'
        groups[row['concept_id']].append(i)
    assert len(index) == 13230 and len(groups) == 1323 and all(len(x) == 10 for x in groups.values())
    n = len(raw)
    errors = np.empty((n, 4, len(ARMS)), np.float64)
    controls = np.empty((n, 4, 2), np.float64)
    risks = np.empty((n, 4, 3), np.float64)
    other = np.roll(np.arange(n).reshape(-1, 4), -1, axis=1).reshape(-1)
    max_phase_error = max_original_error = max_even_odd_error = max_risk_identity_error = 0.
    for lag_index, lag in enumerate([50, 49, 51, 73]):
        old = next(x for x in d33['results'] if x['lag'] == lag)
        target = np.load(check(old['matches']), allow_pickle=False)['target'][:, 0]
        for lo in range(0, n, 128):
            hi = min(n, lo + 128)
            x = segment(raw[lo:hi], lag + 20)[0]
            y, before, after = segment(raw[target[lo:hi]], 20)
            for k, wrong in enumerate((False, True)):
                delta = fit(x, y, before, after, wrong)
                max_phase_error = max(max_phase_error, float(np.max(np.abs(delta - phases[lo:hi, lag_index, k]))))
            controls[lo:hi, lag_index, 0] = one_column_fit((after + before - 2 * y) / 2, x - y, 0, .5)
            controls[lo:hi, lag_index, 1] = one_column_fit(y, x, 0, 1.5)
        for lo in range(0, n, 128):
            hi = min(n, lo + 128)
            x = segment(raw[lo:hi], lag + 90)[0]
            y, before, after = segment(raw[target[lo:hi]], 90)
            delta = np.asarray(phases[lo:hi, lag_index, 0])
            even_delta = np.abs(delta)[:, None, None] * (after + before - 2 * y) / 2
            odd_delta = delta[:, None, None] * (after - before) / 2
            full = predict(y, before, after, delta)
            opposite = predict(y, before, after, -delta)
            max_even_odd_error = max(max_even_odd_error, float(np.max(np.abs(full - (y + even_delta + odd_delta)))))
            preds = [y, full, opposite, predict(y, before, after, phases[other[lo:hi], lag_index, 0]),
                     predict(y, before, after, phases[lo:hi, lag_index, 1], True),
                     y + even_delta, y + odd_delta,
                     y + controls[lo:hi, lag_index, 0, None, None] * (after + before - 2 * y) / 2,
                     controls[lo:hi, lag_index, 1, None, None] * y]
            denominator = (np.square(x).sum((1, 2)) + np.square(y).sum((1, 2))) / 2
            for arm, pred in enumerate(preds):
                errors[lo:hi, lag_index, arm] = np.square(x - pred).sum((1, 2)) / denominator
            r = x - (y + even_delta)
            alignment = 2 * (r * odd_delta).sum((1, 2)) / denominator
            odd_energy = np.square(odd_delta).sum((1, 2)) / denominator
            benefit = errors[lo:hi, lag_index, 5] - errors[lo:hi, lag_index, 1]
            risks[lo:hi, lag_index] = np.column_stack((alignment, odd_energy, benefit))
            max_risk_identity_error = max(max_risk_identity_error, float(np.max(np.abs(benefit - (alignment - odd_energy)))))
            max_original_error = max(max_original_error, float(np.max(np.abs(errors[lo:hi, lag_index, :5] - original_errors[lo:hi, lag_index]))))
            assert time.monotonic() - start < 280
            assert peak_rss() - base_rss < 10 << 30
        print(json.dumps(dict(lag=lag, all_phase_fits_replayed=True, all_held_errors_replayed=True)), flush=True)
    assert max_phase_error < 1e-11 and max_original_error < 1e-11
    assert max_even_odd_error < 1e-10 and max_risk_identity_error < 1e-11
    concept = np.stack([errors.reshape(13230, 4, 4, len(ARMS))[p].mean((0, 1)) for p in groups.values()])
    original_concept = np.load(check(primary['concept_errors']), allow_pickle=False)
    assert np.max(np.abs(concept[:, :, :5] - original_concept)) < 1e-11
    indices = np.random.default_rng(1729).integers(0, 1323, size=(1000, 1323))
    draws = np.concatenate([concept[indices[lo:lo + 20]].mean(1) for lo in range(0, 1000, 20)])
    point = concept.mean(0)
    reductions = (point[:, 0] - point[:, 1]) / point[:, 0]
    reduction_draws = (draws[:, :, 0] - draws[:, :, 1]) / draws[:, :, 0]
    low, high = np.quantile(reduction_draws, [.025, .975], axis=0)
    assert np.max(np.abs(reductions - primary['relative_reductions'])) < 1e-11
    assert np.max(np.abs(low - primary['relative_reductions_descriptive_95_lower'])) < 1e-11
    assert np.max(np.abs(high - primary['relative_reductions_descriptive_95_upper'])) < 1e-11
    original_controls = point[0, [2, 3, 4]] - point[0, 1]
    original_control_draws = draws[:, 0, [2, 3, 4]] - draws[:, 0, 1, None]
    oc_low, oc_high = np.quantile(original_control_draws, [.025, .975], axis=0)
    assert np.max(np.abs(original_controls - primary['lag50_error_benefit_vs_controls'])) < 1e-11
    assert np.max(np.abs(oc_low - primary['lag50_error_benefit_vs_controls_descriptive_95_lower'])) < 1e-11
    assert np.max(np.abs(oc_high - primary['lag50_error_benefit_vs_controls_descriptive_95_upper'])) < 1e-11
    original_lags = reductions[0] - reductions[1:]
    original_lag_draws = reduction_draws[:, 0, None] - reduction_draws[:, 1:]
    ol_low, ol_high = np.quantile(original_lag_draws, [.025, .975], axis=0)
    assert np.max(np.abs(original_lags - primary['lag50_relative_gain_minus_wrong_lags'])) < 1e-11
    assert np.max(np.abs(ol_low - primary['lag50_relative_gain_minus_wrong_lags_descriptive_95_lower'])) < 1e-11
    assert np.max(np.abs(ol_high - primary['lag50_relative_gain_minus_wrong_lags_descriptive_95_upper'])) < 1e-11
    saturation = float((np.abs(phases[:, 0, 0]) >= .5 - 1e-12).mean())
    old_gate = bool(reductions[0] >= .1 and low[0] > 0 and np.all(oc_low > 0)
                    and np.all(ol_low > 0) and saturation < .25)
    assert old_gate == primary['registered_linear_interpolation_phase_explanation_supported']
    matched = [5, 7, 8]
    differences = point[0, matched] - point[0, 1]
    diff_draws = draws[:, 0, matched] - draws[:, 0, 1, None]
    dl, du = np.quantile(diff_draws, [.025, .975], axis=0)
    independent_mechanism = bool(primary['registered_linear_interpolation_phase_explanation_supported']
                                 and np.all(differences / point[0, 0] >= .01) and np.all(dl > 0))
    np.save(output / 'all52920x4x9_held_errors.npy', errors, allow_pickle=False)
    np.save(output / 'all52920x4x2_control_parameters.npy', controls, allow_pickle=False)
    np.save(output / 'all52920x4x3_odd_risk_terms.npy', risks, allow_pickle=False)
    results = dict(status='D34_FULL_RAW_REPLAY_AND_REGISTERED_EVEN_ODD_MECHANISM_CONTROLS_COMPLETE',
                   UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), primary_completion=ref(check(spec['primary_completion'])),
                   source_sha256=sha(__file__), maximum_phase_fit_abs_error=max_phase_error,
                   maximum_original_held_error_abs_error=max_original_error, maximum_even_odd_identity_error=max_even_odd_error,
                   maximum_odd_risk_identity_error=max_risk_identity_error, original_1000_bootstrap_intervals_recomputed=True,
                   original_registered_phase_gate_recomputed_unchanged=old_gate,
                   all423360_original_local_fits_replayed=True, all1058400_original_query_arm_errors_replayed=True,
                   additional_scalar_control_fits=52920 * 4 * 2, arms=ARMS, lags=[50, 49, 51, 73],
                   all_lags_mean_held_errors=point.tolist(), lag50_mean_odd_risk_terms=risks[:, 0].mean(0).tolist(),
                   odd_risk_term_order=['twice_held_residual_alignment','odd_update_energy','even_error_minus_full_error'],
                   lag50_full_benefit_vs_matched_controls=differences.tolist(), matched_control_order=[ARMS[i] for i in matched],
                   lag50_full_benefit_vs_matched_controls_descriptive_95_lower=dl.tolist(),
                   lag50_full_benefit_vs_matched_controls_descriptive_95_upper=du.tolist(),
                   lag50_full_benefit_vs_matched_controls_relative_to_original_error=(differences / point[0, 0]).tolist(),
                   registered_independent_directional_mechanism_gate_passed=independent_mechanism,
                   decision='DIRECTIONAL_INCREMENT_HAS_MATCHED_CONTROL_SUPPORT_STILL_NOT_EVENTS_OR_IMAGES' if independent_mechanism
                   else 'DO_NOT_PROMOTE_PHASE_AS_INDEPENDENT_ARCHITECTURE_MECHANISM',
                   new_neural_models=0, new_SGD_updates=0, new_GPU_jobs=0, new_images=0,
                   D33_or_D34_original_gate_changed=False, normal_inference_neighbor_queries_allowed=False,
                   no_automatic_model_or_image_admission=True, seconds=time.monotonic() - start,
                   peak_extra_CPU_bytes=peak_rss() - base_rss)
    write(output / 'completion.json', results)
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--spec', type=Path, required=True)
    args = p.parse_args()
    run(args.spec)
