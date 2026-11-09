"""Finite T-only relative-latency measurement probe and EEG-front-end export.

This program reads only the SHA-bound physical four-repeat EEG and T index.
It does not import a fitted encoder, visual targets, ordinary, a sampler, or GPU
libraries. Alignment is a classical measurement hypothesis, not a novelty claim.
No result in this diagnostic automatically admits training or image generation.
"""
import os
for _thread_name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_thread_name] = '2'

import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

PROCESS_START = time.monotonic()
import numpy as np
import brainhive_relative_latency_20261028_v1 as M


N = 13230
N_CONCEPTS = 1323
BATCH = 128
OUTPUT_CAP = 600 << 20
EXTRA_RSS_CAP = 2 << 30
TRIALS_SHA = 'b804a105822bfab5eeb853b6942d25109eaf88f3f8a1eb5b503b444a4dc0d393'
INDEX_SHA = 'a12c2758f8555223e08022e5b4a195da37ad3dac37cb6e2806aaf17bac014e67'
EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
PARTITIONS = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))
CHANNEL_A = tuple(range(0, 17, 2))
CHANNEL_B = tuple(range(1, 17, 2))
ARMS = ('zero_shift', 'correct_relative_shift', 'wrong_shift_assignment')
METRICS = ('within_concept_top1', 'within_concept_2AFC', 'correct_minus_other9_cosine')
CORE = slice(12, 238)


def rss_bytes():
    """Peak resident bytes; include mapped input/output pages and temporaries."""
    status = Path('/proc/self/status')
    if status.is_file():
        for line in status.read_text().splitlines():
            if line.startswith('VmHWM:'):
                return int(line.split()[1]) * 1024
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                        ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                        ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
        value = Counters()
        value.cb = ctypes.sizeof(value)
        ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
        ctypes.windll.psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        process = ctypes.windll.kernel32.GetCurrentProcess()
        assert ctypes.windll.psapi.GetProcessMemoryInfo(process, ctypes.byref(value), value.cb)
        return int(value.PeakWorkingSetSize)
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == 'darwin' else 1024)


class Budget:
    def __init__(self, seconds, output):
        self.maximum_seconds = seconds
        self.output = output
        self.base_rss = rss_bytes()
        self.peak_rss = self.base_rss
        self.stage = 'START'

    def check(self, stage=None):
        if stage is not None:
            self.stage = stage
        self.peak_rss = max(self.peak_rss, rss_bytes())
        assert self.peak_rss - self.base_rss < EXTRA_RSS_CAP, 'extra peak RSS exceeds2GiB'
        assert time.monotonic() - PROCESS_START < self.maximum_seconds, 'fixed240second budget exceeded'
        assert not any(name in sys.modules for name in ('torch', 'tensorflow', 'jax', 'cupy'))
        assert self.output_size() < OUTPUT_CAP, 'fixed600MiB output budget exceeded'

    def sha(self, path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as stream:
            while chunk := stream.read(8 << 20):
                digest.update(chunk)
                self.check()
        return digest.hexdigest()

    def ref(self, path):
        path = Path(path)
        return dict(path=str(path.resolve()), sha256=self.sha(path), bytes=path.stat().st_size)

    def checked(self, reference):
        path = Path(reference['path'])
        assert path.is_file() and self.sha(path) == reference['sha256'], str(path)
        if 'bytes' in reference:
            assert path.stat().st_size == reference['bytes']
        return path

    def output_size(self):
        return sum(path.stat().st_size for path in self.output.rglob('*') if path.is_file())

    def write(self, name, value):
        path = self.output / name
        temporary = path.with_name(path.name + '.pending')
        assert not path.exists() and not temporary.exists()
        raw = (json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
        assert self.output_size() + len(raw) < OUTPUT_CAP
        with temporary.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.rename(path)
        self.check()
        return self.ref(path)


def unit_rows(value):
    norms = np.linalg.norm(value, axis=-1, keepdims=True)
    result = np.zeros_like(value)
    np.divide(value, norms, out=result, where=norms > 0)
    return result, int((norms == 0).sum())


def matching_metrics(similarity):
    """All ten stimuli retained; ties half-credit for2AFC and1/k forTop1."""
    assert similarity.shape == (10, 10) and np.isfinite(similarity).all()
    selected = np.diag(similarity)
    maximum = similarity.max(axis=1)
    top1 = (selected == maximum).astype(np.float64) / (similarity == maximum[:, None]).sum(axis=1)
    greater = (selected[:, None] > similarity).sum(axis=1)
    equal = (selected[:, None] == similarity).sum(axis=1)
    afc = (greater + .5 * equal - .5) / 9
    gap = selected - (similarity.sum(axis=1) - selected) / 9
    return np.stack((top1, afc, gap), axis=-1)


def metric_checks():
    identity = matching_metrics(np.eye(10))
    assert np.array_equal(identity, np.ones((10, 3)))
    ties = matching_metrics(np.zeros((10, 10)))
    assert np.allclose(ties, np.array([.1, .5, 0.])[None, :], atol=0, rtol=0)
    shifted = matching_metrics(np.roll(np.eye(10), 1, axis=1))
    assert np.array_equal(shifted[:, 0], np.zeros(10))
    assert np.allclose(shifted[:, 1], 4 / 9)
    assert np.allclose(shifted[:, 2], -1 / 9)
    return dict(status='INDEPENDENT_MATCHING_METRIC_EXPECTATION_CHECKS_PASSED',
                exact_identity_top1_2AFC_gap=[1., 1., 1.], all_tied_top1_2AFC_gap=[.1, .5, 0.],
                wrong_one_hot_top1_2AFC_gap=[0., 4 / 9, -1 / 9], full_denominator_retained=True)


def bootstrap_summary(concept_values, indices, budget):
    value = np.asarray(concept_values, dtype=np.float64)
    assert value.shape[0] == N_CONCEPTS and np.isfinite(value).all()
    flat = value.reshape(N_CONCEPTS, -1)
    draws = np.empty((1000, flat.shape[1]), dtype=np.float64)
    for lo in range(0, 1000, 10):
        draws[lo:lo + 10] = flat[indices[lo:lo + 10]].mean(axis=1)
        budget.check()
    low, high = np.quantile(draws, [.025, .975], axis=0)
    return dict(mean=value.mean(axis=0).tolist(),
                descriptive_95_lower=low.reshape(value.shape[1:]).tolist(),
                descriptive_95_upper=high.reshape(value.shape[1:]).tolist(),
                positive_concepts=(value > 0).sum(axis=0).tolist(),
                zero_concepts=(value == 0).sum(axis=0).tolist(),
                negative_concepts=(value < 0).sum(axis=0).tolist())


def pair_agreement(records, channels):
    """Evaluate all6pairs on exactly the same226 original time coordinates."""
    z = records[:, :, channels, CORE].astype(np.float64).reshape(len(records), 4, -1)
    cosine = np.zeros(len(records), dtype=np.float64)
    mse = np.zeros_like(cosine)
    zero_norms = 0
    for left, right in EDGES:
        numerator = (z[:, left] * z[:, right]).sum(axis=1)
        denominator = np.linalg.norm(z[:, left], axis=1) * np.linalg.norm(z[:, right], axis=1)
        score = np.zeros_like(numerator)
        np.divide(numerator, denominator, out=score, where=denominator > 0)
        cosine += score / 6
        mse += np.square(z[:, left] - z[:, right]).mean(axis=1) / 6
        zero_norms += int((denominator == 0).sum())
    return np.stack((cosine, mse), axis=-1), zero_norms


def boundary_count(tau):
    positions = np.arange(250, dtype=np.float32)[None, None, :] + tau[:, :, None]
    return dict(record_time_positions_outside_0_249=int(((positions < 0) | (positions > 249)).sum()),
                records_nonzero_shift=int((tau != 0).sum()),
                maximum_absolute_tau_samples=float(np.abs(tau).max(initial=0.)))


def save_npy_new(path, value):
    assert not path.exists()
    with path.open('xb') as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-contract', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, required=True)
    args = parser.parse_args()
    assert args.max_seconds == 240, 'Do not expand the fixed CPU budget'
    assert not args.output.exists(), 'Output must be a new directory; preserve failures'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(args.output.parent).free > OUTPUT_CAP + (64 << 20)
    args.output.mkdir()
    budget = Budget(args.max_seconds, args.output)
    try:
        assert tuple(M.EDGES) == EDGES and M.MAX_LAG == 12
        assert M.CORE_START == 12 and M.CORE_STOP == 238
        source_ref = budget.ref(__file__)
        mathematics_module_ref = budget.ref(M.__file__)
        contract_ref = budget.ref(args.input_contract)
        contract = json.loads(args.input_contract.read_bytes())
        assert contract['status'] == 'PHYSICAL_R4_T_INPUTS_READY'
        assert contract['subject'] == 'sub-01' and contract['export_shape'] == [N, 4, 17, 250]
        assert contract['T_rows'] == N and contract['T_concepts'] == N_CONCEPTS
        assert contract['complete_original_mean_replay_rows'] == 16540 and contract['mean_replay_exact'] is True
        assert contract['independent_session_metadata_available'] is False
        trials_ref = contract['files']['trials.npy']
        index_ref = contract['files']['index.jsonl']
        assert trials_ref['sha256'] == TRIALS_SHA and index_ref['sha256'] == INDEX_SHA
        trials_path = budget.checked(trials_ref)
        index_path = budget.checked(index_ref)
        index = [json.loads(line) for line in index_path.read_text(encoding='utf-8').splitlines()]
        assert len(index) == N and all(row['row'] == i for i, row in enumerate(index))
        assert all(row['subject_id'] == 'sub-01' and row['partition'] == 'T' and row['fold'] in (0, 1) for row in index)
        assert len({row['stimulus_id'] for row in index}) == len({row['array_row'] for row in index}) == N
        groups = {}
        for row in index:
            groups.setdefault(row['concept_id'], []).append(row['row'])
        assert len(groups) == N_CONCEPTS and set(map(len, groups.values())) == {10}
        concepts = list(groups)
        folds = np.array([row['fold'] for row in index], dtype=np.int8)
        concept_folds = []
        for concept in concepts:
            groups[concept] = sorted(groups[concept])
            assert len(set(folds[groups[concept]].tolist())) == 1, 'Concept leakage between folds'
            concept_folds.append(int(folds[groups[concept][0]]))
        fold_counts = np.bincount(folds, minlength=2)
        assert fold_counts.tolist() in ([6610, 6620], [6620, 6610])
        metric_ref = budget.write('metric_checks.json', metric_checks())
        trials = np.load(trials_path, mmap_mode='r', allow_pickle=False)
        assert trials.shape == (N, 4, 17, 250) and trials.dtype == np.float32
        budget.check('INPUTS_VERIFIED')

        output_pending = [args.output / (name + '.pending') for name in ('correct_aligned.npy', 'wrong_assignment.npy')]
        outputs = [np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(N, 17, 250))
                   for path in output_pending]
        all_tau = np.empty((N, 4), dtype=np.float32)
        edge_lags = np.empty((N, 6), dtype=np.int16)
        edge_scores = np.empty((N, 6), dtype=np.float32)
        edge_residuals = np.empty((N, 6), dtype=np.float32)
        cross_tau = np.empty((N, 2, 4), dtype=np.float32)
        cross_lags = np.empty((N, 2, 6), dtype=np.int16)
        # [stimulus, estimation-direction, zero/correct/wrong, cosine/MSE].
        cross_values = np.empty((N, 2, 3, 2), dtype=np.float64)
        baseline_sums = np.zeros((2, 17, 250), dtype=np.float64)
        boundary = {arm: dict(record_time_positions_outside_0_249=0, records_nonzero_shift=0,
                             maximum_absolute_tau_samples=0.) for arm in ARMS[1:]}
        lag_boundary_counts = np.zeros((3, 2), dtype=np.int64)
        zero_cross_norms = np.zeros((2, 3), dtype=np.int64)
        maximum_zero_sum_error = 0.
        maximum_edge_cycle_residual = 0.
        for lo in range(0, N, BATCH):
            budget.check('ALIGNMENT_EXPORT_AND_CROSS_ELECTRODE_%d' % lo)
            hi = min(lo + BATCH, N)
            block = np.array(trials[lo:hi], dtype=np.float32, copy=True)
            assert np.isfinite(block).all()
            assert np.any(block[:, 1:] != block[:, :1], axis=(1, 2, 3)).all()
            baseline = block.mean(axis=1, dtype=np.float32)
            for fold in (0, 1):
                baseline_sums[fold] += baseline[folds[lo:hi] == fold].sum(axis=0, dtype=np.float64)
            formal = M.four_align(block, channels=None)
            assert formal['mean'].shape == formal['wrong_mean'].shape == (hi - lo, 17, 250)
            assert np.array_equal(formal['baseline'], baseline)
            assert np.isfinite(formal['mean']).all() and np.isfinite(formal['wrong_mean']).all()
            assert np.array_equal(formal['wrong_tau'], np.roll(formal['tau'], 1, axis=1))
            outputs[0][lo:hi] = formal['mean']
            outputs[1][lo:hi] = formal['wrong_mean']
            all_tau[lo:hi], edge_lags[lo:hi] = formal['tau'], formal['edge_lags']
            edge_scores[lo:hi], edge_residuals[lo:hi] = formal['edge_scores'], formal['edge_residuals']
            maximum_zero_sum_error = max(maximum_zero_sum_error, float(np.abs(formal['tau'].sum(axis=1)).max(initial=0)))
            maximum_edge_cycle_residual = max(maximum_edge_cycle_residual, float(np.abs(formal['edge_residuals']).max(initial=0)))
            lag_boundary_counts[0] += [(formal['edge_lags'] == -12).sum(), (formal['edge_lags'] == 12).sum()]
            for arm, tau in zip(ARMS[1:], (formal['tau'], formal['wrong_tau'])):
                counts = boundary_count(tau)
                for key in ('record_time_positions_outside_0_249', 'records_nonzero_shift'):
                    boundary[arm][key] += counts[key]
                boundary[arm]['maximum_absolute_tau_samples'] = max(boundary[arm]['maximum_absolute_tau_samples'], counts['maximum_absolute_tau_samples'])
            del formal
            for direction, (fit_channels, held_channels) in enumerate(((CHANNEL_A, CHANNEL_B), (CHANNEL_B, CHANNEL_A))):
                estimated = M.estimate_relative_shifts(block, channels=fit_channels)
                cross_tau[lo:hi, direction] = estimated['tau']
                cross_lags[lo:hi, direction] = estimated['edge_lags']
                lag_boundary_counts[direction + 1] += [(estimated['edge_lags'] == -12).sum(), (estimated['edge_lags'] == 12).sum()]
                for arm_index in range(3):
                    if arm_index == 0:
                        shifted = block
                    else:
                        tau = estimated['tau'] if arm_index == 1 else np.roll(estimated['tau'], 1, axis=1)
                        shifted = M.shift_records(block, tau)
                    values, zero_count = pair_agreement(shifted, held_channels)
                    cross_values[lo:hi, direction, arm_index] = values
                    zero_cross_norms[direction, arm_index] += zero_count
                del estimated, shifted
            del block, baseline
            if lo % (BATCH * 16) == 0:
                print(json.dumps(dict(stage=budget.stage, exported_rows=hi, seconds=time.monotonic() - PROCESS_START)), flush=True)
        assert maximum_zero_sum_error < 1e-5
        centers = baseline_sums / fold_counts[:, None, None]
        assert np.isfinite(centers).all()
        for output in outputs:
            output.flush()
            output._mmap.close()
        del outputs
        gc.collect()
        output_paths = []
        for pending in output_pending:
            with pending.open('rb') as stream:
                os.fsync(stream.fileno())
            final = pending.with_name(pending.name.removesuffix('.pending'))
            assert not final.exists()
            pending.rename(final)
            output_paths.append(final)
        budget.check('FORMAL_EXPORT_COMPLETE')

        # Only ten stimuli are resident for the two disjoint pairs. Their lags
        # were estimated from each record-pair alone in the formal6edge pass;
        # reusing them does not read the other pair or any image/target.
        results = np.empty((N_CONCEPTS, 3, 2, 3, 3), dtype=np.float64)
        stimulus_all_directions = np.empty((N, 3, 2, 3, 3), dtype=np.float64)
        stimulus_sums = np.zeros((N, 3, 3), dtype=np.float64)
        stimulus_counts = np.zeros(N, dtype=np.int8)
        pair_zero_norms = np.zeros((3, 3, 2), dtype=np.int64)
        edge_lookup = {edge: i for i, edge in enumerate(EDGES)}
        for ci, concept in enumerate(concepts):
            if ci % 32 == 0:
                budget.check('DISJOINT_RECORD_PAIR_MATCHING_%d' % ci)
            rows = groups[concept]
            block = np.array(trials[rows], dtype=np.float32, copy=True)
            common_center = centers[1 - concept_folds[ci]].reshape(1, -1)
            for partition_index, partition in enumerate(PARTITIONS):
                means_by_pair = []
                for pair in partition:
                    records = np.array(block[:, pair], dtype=np.float32, copy=True)
                    lag = edge_lags[rows, edge_lookup[pair]].astype(np.float32)
                    tau = np.stack((-.5 * lag, .5 * lag), axis=1)
                    means = (records.mean(axis=1, dtype=np.float32),
                             M.shift_records(records, tau).mean(axis=1, dtype=np.float32),
                             M.shift_records(records, tau[:, ::-1]).mean(axis=1, dtype=np.float32))
                    means_by_pair.append(means)
                for arm_index in range(3):
                    left, zero_left = unit_rows(means_by_pair[0][arm_index].reshape(10, -1).astype(np.float64) - common_center)
                    right, zero_right = unit_rows(means_by_pair[1][arm_index].reshape(10, -1).astype(np.float64) - common_center)
                    pair_zero_norms[partition_index, arm_index] += [zero_left, zero_right]
                    similarity = left @ right.T
                    assert np.isfinite(similarity).all() and np.abs(similarity).max() <= 1 + 1e-10
                    for direction, matrix in enumerate((similarity, similarity.T)):
                        metrics = matching_metrics(matrix)
                        results[ci, partition_index, direction, arm_index] = metrics.mean(axis=0)
                        stimulus_all_directions[rows, partition_index, direction, arm_index] = metrics
                        stimulus_sums[rows, arm_index] += metrics
                        if arm_index == 0:
                            stimulus_counts[rows] += 1
            del block, records, means_by_pair, means, left, right, similarity
        assert (stimulus_counts == 6).all() and np.isfinite(results).all()
        stimulus_average = stimulus_sums / 6
        average_directions = np.stack([stimulus_average[groups[concept]].mean(axis=0) for concept in concepts])
        assert np.allclose(average_directions, results.mean(axis=(1, 2)), atol=1e-13, rtol=1e-13)
        cross_concepts = np.stack([cross_values[groups[concept]].mean(axis=0) for concept in concepts])
        budget.check('FIXED_CONCEPT_BOOTSTRAP')
        bootstrap_indices = np.random.default_rng(1729).integers(0, N_CONCEPTS, size=(1000, N_CONCEPTS))
        overall = bootstrap_summary(average_directions, bootstrap_indices, budget)
        all_directions = bootstrap_summary(results, bootstrap_indices, budget)
        primary_differences = bootstrap_summary(np.stack((average_directions[:, 1] - average_directions[:, 0],
                                                         average_directions[:, 1] - average_directions[:, 2]), axis=1),
                                                bootstrap_indices, budget)
        cross_overall = bootstrap_summary(cross_concepts, bootstrap_indices, budget)
        cross_differences = bootstrap_summary(np.stack((cross_concepts[:, :, 1] - cross_concepts[:, :, 0],
                                                       cross_concepts[:, :, 1] - cross_concepts[:, :, 2]), axis=2),
                                              bootstrap_indices, budget)
        budget.check('SAVE_ALL_RESULTS')
        per_concept_path = args.output / 'per_concept.jsonl'
        with per_concept_path.open('x', encoding='utf-8') as stream:
            for ci, concept in enumerate(concepts):
                stream.write(json.dumps(dict(concept_id=concept, fold=concept_folds[ci], stimulus_rows=groups[concept],
                                             recordpair_metrics=results[ci].tolist(), cross_electrode_metrics=cross_concepts[ci].tolist()),
                                        separators=(',', ':'), allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        stimulus_path = args.output / 'per_stimulus_all6_average.npy'
        stimulus_directions_path = args.output / 'per_stimulus_all_directions.npy'
        cross_path = args.output / 'per_stimulus_cross_electrode.npy'
        shifts_path = args.output / 'shifts.npy'
        pair_lags_path = args.output / 'pair_lags.npy'
        save_npy_new(stimulus_path, stimulus_average)
        save_npy_new(stimulus_directions_path, stimulus_all_directions)
        save_npy_new(cross_path, cross_values)
        save_npy_new(shifts_path, all_tau)
        save_npy_new(pair_lags_path, edge_lags)
        measurements_path = args.output / 'latency_measurements.npz'
        with measurements_path.open('xb') as stream:
            np.savez(stream, tau=all_tau, edge_lags=edge_lags, edge_scores=edge_scores, edge_residuals=edge_residuals,
                     cross_electrode_tau=cross_tau, cross_electrode_edge_lags=cross_lags,
                     common_oppositefold_mean4_centers=centers, folds=folds)
            stream.flush()
            os.fsync(stream.fileno())
        # The two complete outputs and every actual input are hashed before a
        # completion summary exists. Partial arrays never imply completion.
        exports = {name: budget.ref(path) for name, path in zip(('correct_aligned', 'wrong_assignment'), output_paths)}
        budget.checked(trials_ref)
        budget.checked(index_ref)
        budget.checked(contract_ref)
        budget.checked(source_ref)
        budget.checked(mathematics_module_ref)
        budget.check('FINAL_SHA_AND_RESOURCE_CHECKS')
        summary = dict(
            status='D28_COMPLETE_T_RELATIVE_LATENCY_MEASUREMENT_CPU',
            source=source_ref, mathematics_module=mathematics_module_ref, input_contract=contract_ref,
            trials=trials_ref, index=index_ref,
            inputs_actually_read=[trials_ref, index_ref], subject='sub-01', T_stimuli=N, T_concepts=N_CONCEPTS,
            folds=fold_counts.tolist(), original_physical_repeats=4, input_shape=[N, 4, 17, 250],
            output_shape=[N, 17, 250], output_dtype='float32', aligned_mean4_exports=exports,
            correct_aligned=exports['correct_aligned'], wrong_assignment=exports['wrong_assignment'],
            aligned_EEG=exports['correct_aligned'], wrong_EEG=exports['wrong_assignment'],
            shifts=budget.ref(shifts_path), pair_lags=budget.ref(pair_lags_path),
            compact_row_order='Every array has the unchanged index.jsonl compactrow0..13229 in exactly that order',
            zero_shift_baseline='Original four-repeat arithmetic mean; recomputable from the unchanged SHA-bound raw trials',
            metadata_timebase=dict(sampling_rate_Hz=250, points=250, maximum_edge_lag_samples=12,
                                   maximum_edge_lag_ms=48, source='Existing released250Hz metadata, no new event reconstruction'),
            lag_measurement=dict(edges=EDGES, score_lags=list(range(-12, 13)),
                                 fixed_correlation_center_timepoints=[12, 238],
                                 score_formula='0.5*(corr(xa_core,xb_shift(+d))+corr(xb_core,xa_shift(-d))); each electrode time-centered per window, pooled normalized correlation',
                                 tie_rule='Prefer minimum absolute lag among exact maxima; if both+/-d share that minimum absolute maximum, abstain to0',
                                 estimator='Six independent pairwise normalized correlations; zero-sum graph least-squares tau=B.T@d/4',
                                 channel_scope='All17 for the exported means and independent recordpair matching',
                                 edge_boundary_counts_axes=('all17_or_A_or_B', 'minus12_or_plus12'),
                                 edge_boundary_counts=lag_boundary_counts.tolist(),
                                 maximum_tau_sum_absolute_error_samples=maximum_zero_sum_error,
                                 maximum_absolute_edge_fit_residual_samples=maximum_edge_cycle_residual),
            interpolation='Each record uses x(t+tau); fixed original250 time coordinates, linear interpolation of the zero-extended discrete sample grid; no circular wrap',
            wrong_control='Formal4record tau assigned with np.roll(tau,1,axis=1); pairinternal2record tau swapped, i.e. opposite alignment',
            formal_boundary_counts=boundary,
            cross_electrode=dict(direction_names=('A_estimate_B_evaluate', 'B_estimate_A_evaluate'),
                                 channel_A=CHANNEL_A, channel_B=CHANNEL_B, arms=ARMS,
                                 evaluation_metrics=('mean_of6pair_flattened_cosine', 'mean_of6pair_MSE'),
                                 evaluation_timepoints=[12, 238], centered_by_an_arm_fit=False,
                                 per_concept_average_then_bootstrap=cross_overall,
                                 differences=dict(axes=('direction', 'correct_minus_zero_or_correct_minus_wrong', 'metric'), **cross_differences),
                                 zero_norm_pair_counts=zero_cross_norms.tolist(),
                                 claim_limit='Functional cross-electrode measurement diagnostic; volume-conducted noise and artifacts can generalize across held channels. No physiological causality is established.'),
            disjoint_recordpair_matching=dict(partitions=PARTITIONS, directions=('left_to_right', 'right_to_left'), arms=ARMS, metrics=METRICS,
                                             pair_shift_rule='lag from that record-pair only; tau=(-lag/2,+lag/2), independent of the other pair',
                                             common_center='One original mean4 center fitted on the opposite concept fold; same center for both pairs and every arm',
                                             gallery='All10same-concept stimuli; correct stimulus identity used only to score matching',
                                             tie_policy='Top1 correct in k-way max tie gets1/k; 2AFC tie gets0.5; zero-norm vectors get0cosine; no denominator exclusions',
                                             zero_norm_features=pair_zero_norms.tolist(),
                                             all_fixed_directions=dict(axes=('partition', 'direction', 'arm', 'metric'), **all_directions),
                                             average_all3_partitions_and2_directions=dict(axes=('arm', 'metric'), **overall)),
            primary_readout=dict(contrast_order=('correct_minus_zero', 'correct_minus_wrong'), metric_order=METRICS,
                                 concept_differences=primary_differences,
                                 interpretation='Review these prespecified differences in2AFC andTop1; no automatic model or image admission'),
            aggregation='Average all3partitions x2directions within each stimulus, then10stimuli perconcept, then1323concepts. Shared four records are never counted as6independent replications.',
            bootstrap=dict(seed=1729, repetitions=1000, unit='whole1323concepts', shared_indices_for_all_contrasts=True,
                           confidence_intervals_descriptive=True, independent_confirmation=False),
            per_concept=budget.ref(per_concept_path), per_stimulus_all6_average=budget.ref(stimulus_path),
            per_stimulus_all_directions=budget.ref(stimulus_directions_path),
            per_stimulus_all_directions_axes=('stimulus', 'partition', 'direction', 'arm', 'metric'),
            per_stimulus_cross_electrode=budget.ref(cross_path), latency_measurements=budget.ref(measurements_path), metric_checks=metric_ref,
            peak_extra_RSS_bytes=budget.peak_rss - budget.base_rss, peak_RSS_bytes=budget.peak_rss,
            maximum_extra_RSS_bytes=EXTRA_RSS_CAP, maximum_output_bytes=OUTPUT_CAP, maximum_seconds=args.max_seconds,
            thread_environment={name: os.environ[name] for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
            seconds=time.monotonic() - PROCESS_START, new_models=0, training_updates=0, new_training_updates=0,
            new_images=0, GPU_imported=False, GPU_initialized=False, CUDA_initialized=False,
            automatic_image_queue=False, ordinary_changed=False, sampler_changed=False, formal_D_resumed=False,
            target_F_c_or_image_arrays_read=False, image_gain_established=False, novelty_established=False,
            independent_session_metadata_available=False, iid_noise_established=False, physiological_latency_established=False,
            information_beyond_c_established=False,
            limitations=['This single-subject completeTprobe is exploratory; released preprocessing lineage remains limited.',
                         'Relative shifts may fit common noise, ERP amplitude changes, or overlapping components; zero-sum fixes a gauge only.',
                         'Cross-electrode and disjoint-record reproducibility do not establish neural origin, causal physiology, increment beyondc, or image benefit.',
                         'Whole-trial shift assumes coherent timing variation; correlations alone cannot establish this assumption.',
                         'Classical alignment and linear folding intoWare not claimed as original contributions.'])
        summary_ref = budget.write('summary.json', summary)
        budget.check('COMPLETE')
        print(json.dumps(dict(status=summary['status'], summary=summary_ref, seconds=time.monotonic() - PROCESS_START,
                              output_bytes=budget.output_size(), peak_extra_RSS_bytes=budget.peak_rss - budget.base_rss)), flush=True)
    except BaseException as exc:
        failure = dict(status='FAILED_FINITE_TONLY_CPU_RELATIVE_LATENCY_PROBE_NO_AUTOMATIC_RETRY', stage=budget.stage,
                       error=repr(exc), traceback=traceback.format_exc(), seconds=time.monotonic() - PROCESS_START,
                       peak_extra_RSS_bytes=budget.peak_rss - budget.base_rss, output_bytes=budget.output_size(),
                       no_automatic_retry=True, new_images=0, new_training_updates=0, ordinary_changed=False, formal_D_resumed=False)
        path = args.output / 'failure.json'
        if not path.exists():
            path.write_text(json.dumps(failure, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
