"""Independent numerical checks for the fixed relative-latency operator."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
import numpy as np
import brainhive_relative_latency_20261028_v1 as V


def _brute_scores(x, channels=None):
    channels = np.arange(17) if channels is None else np.asarray(channels)
    out = np.zeros((len(x), 6, 25), dtype=np.float64)
    def corr(left, right):
        left = left.astype(np.float64)
        right = right.astype(np.float64)
        left -= left.mean(axis=-1, keepdims=True)
        right -= right.mean(axis=-1, keepdims=True)
        norm = np.linalg.norm(left) * np.linalg.norm(right)
        return (left * right).sum() / norm if norm > 0. else 0.
    for batch in range(len(x)):
        for edge, (a, b) in enumerate(V.EDGES):
            for k, lag in enumerate(range(-12, 13)):
                forward = corr(x[batch, a, channels, 12:238], x[batch, b, channels, 12 + lag:238 + lag])
                reverse = corr(x[batch, b, channels, 12:238], x[batch, a, channels, 12 - lag:238 - lag])
                out[batch, edge, k] = .5 * (forward + reverse)
    return out


def _state_equal(a, b):
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=Path('audit/relative_latency_20261028_v1/local_math_checks.json'))
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    global_state = np.random.get_state()
    rng = np.random.RandomState(1729)
    checks = {}
    diagnostics = {}

    def check(name, value):
        if not bool(value):
            raise AssertionError(name)
        checks[name] = True

    x = rng.standard_normal((3, 4, 17, 250)).astype(np.float32)
    original = x.copy()
    estimate = V.estimate_relative_shifts(x)
    brute = _brute_scores(x)
    error = float(np.max(np.abs(estimate['scores'].astype(np.float64) - brute)))
    diagnostics['FFT_vs_independent_float64_brute_max_abs'] = error
    check('FFT_all6edges_all25lags_matches_direct_float64_definition', error < 5e-8)
    subset = [0, 2, 7, 12, 16]
    subset_error = float(np.max(np.abs(V.estimate_relative_shifts(x, subset)['scores'].astype(np.float64) - _brute_scores(x, subset))))
    diagnostics['subset_FFT_vs_brute_max_abs'] = subset_error
    check('channel_subset_matches_independent_definition', subset_error < 5e-8)
    forward64 = V._scores(x, ((0, 1),), None)[:, 0]
    reverse64 = V._scores(x, ((1, 0),), None)[:, 0]
    check('symmetric_scores_are_float64_exact_under_edge_reversal', np.array_equal(forward64, reverse64[:, ::-1]))
    forward_pair = V.estimate_pair_lag(x[:, 0], x[:, 1])
    reverse_pair = V.estimate_pair_lag(x[:, 1], x[:, 0])
    check('random_pair_lag_reverses_sign_when_records_are_exchanged',
          np.array_equal(forward_pair['lag'], -reverse_pair['lag']))

    # A compact-support waveform avoids a boundary-driven synthetic result.
    base = rng.standard_normal((2, 1, 17, 250)).astype(np.float32)
    base[..., :40] = 0.
    base[..., 210:] = 0.
    base = np.repeat(base, 4, axis=1)
    expected_tau = np.array([[-3., -1., 1., 3.], [-4., 0., 1., 3.]], dtype=np.float32)
    delayed = V.shift_records(base, -expected_tau)
    got = V.estimate_relative_shifts(delayed)
    expected_edges = expected_tau @ V.INCIDENCE.T
    check('synthetic_integer_delays_correct_edge_sign_and_zero_sum_tau',
          np.array_equal(got['edge_lags'], expected_edges) and np.array_equal(got['tau'], expected_tau))
    registered = V.shift_records(delayed, got['tau'])
    check('x_t_plus_tau_reconstructs_clear_synthetic_waveform', np.array_equal(registered[..., 20:230], base[..., 20:230]))
    pair = delayed[:, (0, 1)]
    pg = V.pair_align(pair)
    expected_d = expected_tau[:, 1] - expected_tau[:, 0]
    check('physical_pair_alignment_has_correct_plus_minus_half_lag',
          np.array_equal(pg['lag'], expected_d) and np.array_equal(pg['tau'].sum(1), np.zeros(2, np.float32)))
    check('physical_pair_shared_centre_and_waveform_match', np.array_equal(pg['aligned'][:, 0, :, 20:230], pg['aligned'][:, 1, :, 20:230]))

    tau = estimate['tau']
    residual = estimate['edge_residuals']
    check('complete_graph_LS_has_exact_zero_sum_and_normal_equations',
          np.array_equal(tau.sum(1), np.zeros(3, np.float32)) and np.max(np.abs(residual @ V.INCIDENCE)) == 0.)
    perturbed = tau + np.array([[.25, -.25, .5, -.5]], np.float32)
    check('zero_sum_LS_minimises_squared_edge_residual',
          np.all(np.sum(residual ** 2, 1) <= np.sum((perturbed @ V.INCIDENCE.T - estimate['edge_lags']) ** 2, 1)))
    offsets = rng.uniform(-20., 20., (3, 4, 17, 1)).astype(np.float32)
    dc = V.estimate_relative_shifts(x + offsets)
    check('electrode_DC_offsets_do_not_change_clear_lag_choices', np.array_equal(dc['edge_lags'], estimate['edge_lags']))

    # Check every repeat order both for identified shifts and noisy or broad peaks.
    broad = np.cumsum(rng.standard_normal((3, 4, 17, 250)), axis=-1).astype(np.float32)
    broad_estimate = V.estimate_relative_shifts(broad)
    permutation_ok = True
    synthetic_permutation_ok = True
    for perm_tuple in itertools.permutations(range(4)):
        perm = np.array(perm_tuple)
        permutation_ok &= np.array_equal(V.estimate_relative_shifts(x[:, perm])['tau'], estimate['tau'][:, perm])
        permutation_ok &= np.array_equal(V.estimate_relative_shifts(broad[:, perm])['tau'], broad_estimate['tau'][:, perm])
        synthetic_permutation_ok &= np.array_equal(V.estimate_relative_shifts(delayed[:, perm])['tau'], expected_tau[:, perm])
    check('all24_repeat_permutations_equivariant_for_random_noise_and_broad_peaks', permutation_ok)
    check('all24_repeat_permutations_equivariant_for_identifiable_shifts', synthetic_permutation_ok)
    common_delay = np.full((2, 4), 2., np.float32)
    translated = V.estimate_relative_shifts(V.shift_records(delayed, -common_delay))
    check('common_temporal_translation_gauge_preserves_relative_clear_shifts', np.array_equal(translated['tau'], expected_tau))

    zeros = np.zeros_like(x)
    check('zero_signal_tie_selects_zero_lag', np.all(V.estimate_relative_shifts(zeros)['edge_lags'] == 0))
    tied = np.full((1, 25), -1., np.float64)
    tied[0, np.isin(V.LAGS, (-3, 3, -4, 4))] = 1.
    tied_lag, _ = V._best_lags(tied)
    check('opposite_sign_equal_absolute_maxima_abstain_to_zero', int(tied_lag[0]) == 0)
    asymmetric_tie = np.full((1, 25), -1., np.float64)
    asymmetric_tie[0, np.isin(V.LAGS, (-3, 4))] = 1.
    unique_lag, _ = V._best_lags(asymmetric_tie)
    unique_reverse, _ = V._best_lags(asymmetric_tie[:, ::-1])
    check('unique_minimum_absolute_maximum_is_sign_neutral', int(unique_lag[0]) == -3 and int(unique_reverse[0]) == 3)
    check('fully_ambiguous_score_tie_abstains_to_zero', np.all(V._best_lags(np.ones((2, 25), np.float64))[0] == 0))
    identical = np.repeat(x[:, :1], 4, axis=1)
    check('identical_records_select_exact_zero_tau', np.array_equal(V.estimate_relative_shifts(identical)['tau'], np.zeros((3, 4), np.float32)))
    constant = np.full_like(x, 7.3)
    check('constant_electrode_signal_has_exact_zero_scores_and_lags',
          np.all(V.estimate_relative_shifts(constant)['scores'] == 0.) and np.all(V.estimate_relative_shifts(constant)['edge_lags'] == 0))
    wrong = V.cyclic_wrong_tau(tau)
    check('cyclic_wrong_control_preserves_tau_multiset_and_zero_sum',
          np.array_equal(np.sort(wrong, axis=1), np.sort(tau, axis=1)) and np.array_equal(wrong.sum(1), tau.sum(1)))
    zero_tau = np.zeros((3, 4), np.float32)
    check('zero_shift_returns_original_array_object_and_values', V.shift_records(x, zero_tau) is x)
    mixed_tau = zero_tau.copy()
    mixed_tau[0, 1] = .5
    mixed = V.shift_records(x, mixed_tau)
    check('zero_shift_records_in_mixed_batch_are_bitwise_preserved', np.array_equal(mixed[mixed_tau == 0.], x[mixed_tau == 0.]))

    ramp = np.broadcast_to(np.arange(250, dtype=np.float32), (1, 2, 17, 250)).copy()
    interpolation = V.shift_records(ramp, np.array([[.5, -.5]], np.float32))
    check('linear_interpolation_and_zero_extension_edges',
          interpolation[0, 0, 0, 0] == .5 and interpolation[0, 0, 0, -1] == 124.5
          and interpolation[0, 1, 0, 0] == 0. and interpolation[0, 1, 0, -1] == 248.5)
    output = V.four_align(x)
    check('all_public_data_outputs_float32_and_native_baseline_exact',
          output['mean'].dtype == np.float32 and output['wrong_mean'].dtype == np.float32
          and np.array_equal(output['baseline'], x.mean(axis=1, dtype=np.float32)))
    check('operator_does_not_modify_input', np.array_equal(x, original))

    bad_nan = x.copy()
    bad_nan[0, 0, 0, 0] = np.nan
    rejected = []
    for call in (lambda: V.estimate_relative_shifts(bad_nan),
                 lambda: V.estimate_relative_shifts(x[:, :, :, :-1]),
                 lambda: V.estimate_relative_shifts(x.astype(np.float64)),
                 lambda: V.estimate_relative_shifts(x, [0, 0]),
                 lambda: V.shift_records(x, np.full((3, 4), np.nan, np.float32))):
        try:
            call()
        except (ValueError, TypeError):
            rejected.append(True)
    check('rejects_nonfinite_wrong_shape_dtype_channels_and_tau', len(rejected) == 5)
    first = V.estimate_relative_shifts(x)
    second = V.estimate_relative_shifts(x)
    check('deterministic_no_global_NumPy_RNG_consumption',
          _state_equal(global_state, np.random.get_state()) and all(np.array_equal(first[k], second[k]) for k in first))
    check('no_GPU_framework_imported', 'torch' not in sys.modules and 'cupy' not in sys.modules)

    sources = {}
    for filename in ('brainhive_relative_latency_20261028_v1.py', Path(__file__).name):
        path = Path(__file__).resolve().parent / filename
        sources[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    value = dict(status='RELATIVE_LATENCY_FIXED_OPERATOR_CPU_CHECKS_PASSED', checks=checks,
                 diagnostics=diagnostics, source_sha256=sources,
                 fixed=dict(max_lag=12, core_start=12, core_stop_exclusive=238,
                            core_length=226, FFT_length=512,
                            score='symmetric_mean_of_two_fixed_core_directional_correlations',
                            tie='unique_smallest_absolute_maximum_else_zero_for_plus_minus_tie'),
                 limitations=['Classical registration, not an originality claim.',
                              'The zero-sum gauge does not preserve each physiological component latency.'],
                 new_models=0, new_images=0, GPU_used=False, ordinary_changed=False,
                 elapsed_seconds=time.perf_counter() - started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(value, allow_nan=False))


if __name__ == '__main__':
    main()
