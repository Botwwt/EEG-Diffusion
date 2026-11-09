"""T-only disjoint-repeat spectral-structure falsification probe, CPU only.

All four prespecified views are reported: raw/trace-normalized band covariance,
each with full Hermitian entries or diagonal power only.  No image, F, c, fitted
encoder, generator or GPU library is imported.  A synthetic counterexample
demonstrates that reproducible structure need not be induced neural activity.
"""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '2'
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

PROCESS_START = time.monotonic()
import numpy as np


BANDS = ((0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 126))
PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
PARTITIONS = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))
VARIANTS = ('raw_covariance', 'trace_normalized_covariance')
SCOPES = ('full_Hermitian', 'diagonal_power_only')
CONDITIONS = ('correct_pairing', 'cyclic_wrong_pairing', 'correct_minus_cyclic_wrong')
METRICS = ('within_concept_top1', 'within_concept_2AFC', 'correct_minus_other9_cosine')
UPPER = np.triu_indices(17, 1)
FEATURES_PER_BAND = 17 + 2 * len(UPPER[0])
FEATURE_DIM = 6 * FEATURES_PER_BAND
DIAGONAL_INDICES = np.concatenate([np.arange(b * FEATURES_PER_BAND, b * FEATURES_PER_BAND + 17)
                                   for b in range(6)])
ONE_SIDED_WEIGHTS = np.full(126, 2., dtype=np.float64)
ONE_SIDED_WEIGHTS[[0, -1]] = 1.
OUTPUT_CAP = 20 << 20
EXTRA_RSS_CAP = 2 << 30


def rss_bytes():
    """Peak resident pages, including transient allocations; no psutil/torch."""
    status = Path('/proc/self/status')
    if status.is_file():
        for line in status.read_text().splitlines():
            if line.startswith('VmHWM:'):
                return int(line.split()[1]) * 1024
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
        assert self.peak_rss - self.base_rss < EXTRA_RSS_CAP, 'extra RSS exceeds2GiB'
        assert time.monotonic() - PROCESS_START < self.maximum_seconds, 'fixed240second budget exceeded'
        assert 'torch' not in sys.modules and 'tensorflow' not in sys.modules

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

    def checked(self, ref):
        path = Path(ref['path'])
        assert path.is_file() and self.sha(path) == ref['sha256'], str(path)
        if 'bytes' in ref:
            assert path.stat().st_size == ref['bytes']
        return path

    def output_size(self):
        return sum(p.stat().st_size for p in self.output.rglob('*') if p.is_file())

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


def hermitian_vector(matrix):
    assert matrix.shape[-2:] == (17, 17)
    return np.concatenate((np.diagonal(matrix, axis1=-2, axis2=-1).real,
                           np.sqrt(2) * matrix[..., UPPER[0], UPPER[1]].real,
                           np.sqrt(2) * matrix[..., UPPER[0], UPPER[1]].imag), axis=-1)


def feature_pair(x, pair, band_indices):
    """Return both fixed views of the weighted one-sided band spectrum.

    rFFT uses NumPy's default backward normalization, no taper or demeaning.
    DC/Nyquist weights are1, interior weights2; normalize by sum of band weights.
    Trace normalization is per band; exactly zero trace returns a zero vector.
    """
    difference = (x[:, pair[0]] - x[:, pair[1]]) / np.sqrt(2.)
    fourier = np.fft.rfft(difference, n=250, axis=-1)
    raw = np.empty((len(x), FEATURE_DIM), dtype=np.float32)
    normalized = np.empty_like(raw)
    zero_trace_bands = 0
    for bi, bins in enumerate(band_indices):
        z = fourier[..., bins]
        weights = ONE_SIDED_WEIGHTS[bins]
        weighted = z * np.sqrt(weights)
        covariance = np.matmul(weighted, np.swapaxes(weighted.conj(), -1, -2)) / weights.sum()
        vector = hermitian_vector(covariance)
        trace = np.diagonal(covariance, axis1=-2, axis2=-1).real.sum(axis=-1)
        assert np.isfinite(vector).all() and np.isfinite(trace).all() and np.all(trace >= 0)
        normalized_vector = np.zeros_like(vector)
        np.divide(vector, trace[:, None], out=normalized_vector, where=trace[:, None] > 0)
        sl = slice(bi * FEATURES_PER_BAND, (bi + 1) * FEATURES_PER_BAND)
        raw[:, sl] = vector
        normalized[:, sl] = normalized_vector
        zero_trace_bands += int((trace == 0).sum())
    assert np.isfinite(raw).all() and np.isfinite(normalized).all()
    return raw, normalized, zero_trace_bands


def unit_rows(value):
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    result = np.zeros_like(value)
    np.divide(value, norms, out=result, where=norms > 0)
    return result, int((norms == 0).sum())


def matching_metrics(similarity, target_indices):
    """Keep all queries: 2AFC ties half credit; correct in a k-way Top1 tie 1/k."""
    assert similarity.shape == (10, 10) and target_indices.shape == (10,)
    selected = similarity[np.arange(10), target_indices]
    maximum = similarity.max(axis=1)
    top1 = (selected == maximum).astype(np.float64) / (similarity == maximum[:, None]).sum(axis=1)
    greater = (selected[:, None] > similarity).sum(axis=1)
    equal = (selected[:, None] == similarity).sum(axis=1)
    afc = (greater + .5 * equal - .5) / 9
    gap = selected - (similarity.sum(axis=1) - selected) / 9
    return np.stack((top1, afc, gap), axis=-1)


def synthetic_checks(band_indices):
    """Independent algebra tests and a non-neural success counterexample."""
    rng = np.random.default_rng(1729)
    z = rng.normal(size=(17, 11)) + 1j * rng.normal(size=(17, 11))
    covariance = z @ z.conj().T / z.shape[1]
    vector = hermitian_vector(covariance)
    frobenius2 = float(np.square(np.abs(covariance)).sum())
    vector2 = float(vector @ vector)
    assert np.isclose(frobenius2, vector2, atol=1e-10, rtol=1e-12)
    restored = np.diag(vector[:17]).astype(np.complex128)
    restored[UPPER] = (vector[17:153] + 1j * vector[153:]) / np.sqrt(2)
    restored[(UPPER[1], UPPER[0])] = restored[UPPER].conj()
    assert np.allclose(covariance, restored, atol=1e-12, rtol=1e-12)

    # Every stimulus has identical zero mean4. A repeat-varying spatial artifact
    # has stimulus-specific orientation. It yields reproducible covariance in
    # disjoint repeat pairs without any phase-locked signal or neural cause.
    t = np.arange(250) / 250.
    waveform = np.cos(2 * np.pi * 12 * t)
    x = np.zeros((10, 4, 17, 250), dtype=np.float64)
    amplitudes = np.array([-3., -1., 1., 3.])
    for stimulus in range(10):
        x[stimulus, :, stimulus] = amplitudes[:, None] * waveform
    assert np.allclose(x.mean(axis=1), 0, atol=1e-14)
    pair_features = {}
    for pair in PAIRS:
        raw, normalized, _ = feature_pair(x, pair, band_indices)
        reversed_raw, reversed_normalized, _ = feature_pair(x, pair[::-1], band_indices)
        assert np.array_equal(raw, reversed_raw) and np.array_equal(normalized, reversed_normalized)
        pair_features[pair] = (raw, normalized)
    counterexample = []
    for partition in PARTITIONS:
        # Trace-normalized 8-16Hz band avoids insignificant FFT leakage bins.
        sl = slice(2 * FEATURES_PER_BAND, 3 * FEATURES_PER_BAND)
        left = pair_features[partition[0]][1][:, sl].astype(np.float64)
        right = pair_features[partition[1]][1][:, sl].astype(np.float64)
        left -= left.mean(axis=0)
        right -= right.mean(axis=0)
        left, _ = unit_rows(left)
        right, _ = unit_rows(right)
        similarity = left @ right.T
        metrics = matching_metrics(similarity, np.arange(10))
        assert metrics[:, 0].mean() == metrics[:, 1].mean() == 1.
        counterexample.append(dict(partition=partition, within_concept_top1=float(metrics[:, 0].mean()),
                                   within_concept_2AFC=float(metrics[:, 1].mean())))
    common = rng.normal(size=(2, 1, 17, 250))
    common = np.repeat(common, 4, axis=1)
    raw, normalized, zero = feature_pair(common, (0, 1), band_indices)
    assert np.count_nonzero(raw) == np.count_nonzero(normalized) == 0 and zero == 12
    return dict(status='SYNTHETIC_ALGEBRA_AND_NON_NEURAL_COUNTEREXAMPLE_CHECKS_PASSED',
                Hermitian_Frobenius_squared_norm=frobenius2, vector_squared_norm=vector2,
                squared_norm_absolute_error=abs(frobenius2 - vector2),
                inverse_Hermitian_reconstruction=True, pair_sign_reversal_exactly_invariant=True,
                identical_four_repeats_cancel_in_difference=True,
                zero_band_trace_returns_zero_vector=True,
                non_neural_artifact_identical_mean4_counterexample=counterexample,
                counterexample_implication='Perfect disjoint-pair matching can arise from a stimulus-dependent repeat-varying artifact with zero mean4. Reproducibility alone does not establish neural origin, induced activity, or image-useful information.')


def bootstrap_summary(concept_values, indices):
    """Bounded batches; never construct [1000,1323,all_metrics] tensors."""
    data = np.asarray(concept_values, dtype=np.float64)
    assert data.shape[0] == 1323
    flat = data.reshape(1323, -1)
    draws = np.empty((1000, flat.shape[1]), dtype=np.float64)
    for lo in range(0, 1000, 10):
        draws[lo:lo + 10] = flat[indices[lo:lo + 10]].mean(axis=1)
    low, high = np.quantile(draws, [.025, .975], axis=0)
    return dict(mean=data.mean(axis=0).tolist(),
                descriptive_95_lower=low.reshape(data.shape[1:]).tolist(),
                descriptive_95_upper=high.reshape(data.shape[1:]).tolist(),
                positive_concepts=(data > 0).sum(axis=0).tolist(),
                zero_concepts=(data == 0).sum(axis=0).tolist(),
                negative_concepts=(data < 0).sum(axis=0).tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-contract', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, required=True)
    args = parser.parse_args()
    assert args.max_seconds == 240, 'Do not silently expand the fixed CPU budget'
    assert not args.output.exists()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(args.output.parent).free > 25 << 20
    args.output.mkdir()
    budget = Budget(args.max_seconds, args.output)
    try:
        contract_ref = budget.ref(args.input_contract)
        contract = json.loads(args.input_contract.read_bytes())
        assert contract['status'] == 'PHYSICAL_R4_T_INPUTS_READY'
        assert contract['subject'] == 'sub-01' and contract['export_shape'] == [13230, 4, 17, 250]
        assert contract['T_rows'] == 13230 and contract['T_concepts'] == 1323
        assert contract['complete_original_mean_replay_rows'] == 16540 and contract['mean_replay_exact'] is True
        assert contract['independent_session_metadata_available'] is False
        trials_ref = contract['files']['trials.npy']
        index_ref = contract['files']['index.jsonl']
        trials_path = budget.checked(trials_ref)
        index_path = budget.checked(index_ref)
        index = [json.loads(line) for line in index_path.read_text(encoding='utf-8').splitlines()]
        assert len(index) == 13230 and all(r['row'] == i for i, r in enumerate(index))
        assert all(r['subject_id'] == 'sub-01' and r['partition'] == 'T' and r['fold'] in (0, 1) for r in index)
        assert len({r['stimulus_id'] for r in index}) == len({r['array_row'] for r in index}) == 13230
        groups = {}
        for r in index:
            groups.setdefault(r['concept_id'], []).append(r['row'])
        assert len(groups) == 1323 and set(map(len, groups.values())) == {10}
        concepts = list(groups)
        folds = np.array([r['fold'] for r in index], dtype=np.int8)
        concept_folds = []
        for concept in concepts:
            group = groups[concept]
            assert len(set(folds[group].tolist())) == 1, 'Concept leakage between folds'
            groups[concept] = sorted(group)
            concept_folds.append(int(folds[group[0]]))
        fold_counts = np.bincount(folds, minlength=2)
        assert fold_counts.tolist() in ([6610, 6620], [6620, 6610])
        frequencies = np.fft.rfftfreq(250, d=1 / 250.)
        band_indices = [np.flatnonzero((frequencies >= low) & (frequencies < high)) for low, high in BANDS]
        assert list(map(len, band_indices)) == [4, 4, 8, 16, 32, 62]
        assert np.array_equal(np.concatenate(band_indices), np.arange(126))
        synthetic = synthetic_checks(band_indices)
        synthetic_ref = budget.write('synthetic_checks.json', synthetic)
        budget.check('INPUTS_AND_SYNTHETIC_CHECKS_VERIFIED')
        trials = np.load(trials_path, mmap_mode='r', allow_pickle=False)
        assert trials.shape == (13230, 4, 17, 250) and trials.dtype == np.float32
        # [concept, raw/trace, full/diagonal, partition, direction, condition, metric]
        results = np.empty((1323, 2, 2, 3, 2, 3, 3), dtype=np.float64)
        stimulus_sums = np.zeros((13230, 2, 2, 3, 3), dtype=np.float64)
        stimulus_counts = np.zeros((13230, 2, 2), dtype=np.int8)
        zero_features = np.zeros((2, 2, 3, 2), dtype=np.int64)
        zero_traces = np.zeros((3, 2), dtype=np.int64)
        feature_stats = []
        for pi, partition in enumerate(PARTITIONS):
            budget.check('PARTITION_%d_FEATURES' % pi)
            # Only this partition's two pairs are resident. Full float64 feature
            # libraries or centered copies of all13,230 stimuli are prohibited.
            raw_features = np.empty((2, 13230, FEATURE_DIM), dtype=np.float32)
            norm_features = np.empty_like(raw_features)
            sums = np.zeros((2, 2, 2, FEATURE_DIM), dtype=np.float64)
            for lo in range(0, 13230, 256):
                budget.check()
                hi = min(lo + 256, 13230)
                block = np.array(trials[lo:hi], dtype=np.float32, copy=True)
                assert np.isfinite(block).all()
                assert np.any(block[:, 1:] != block[:, :1], axis=(1, 2, 3)).all()
                for pair_i, pair in enumerate(partition):
                    raw, normalized, zero = feature_pair(block, pair, band_indices)
                    raw_features[pair_i, lo:hi] = raw
                    norm_features[pair_i, lo:hi] = normalized
                    zero_traces[pi, pair_i] += zero
                    for variant_i, features in enumerate((raw, normalized)):
                        for fold in (0, 1):
                            sums[variant_i, pair_i, fold] += features[folds[lo:hi] == fold].sum(axis=0, dtype=np.float64)
                del block, raw, normalized
            centers = sums / fold_counts[None, None, :, None]
            assert np.isfinite(centers).all()
            feature_stats.append(dict(partition=partition,
                                      center_fold_counts=fold_counts.tolist(),
                                      opposite_fold_center_norms=np.linalg.norm(centers, axis=-1).tolist(),
                                      zero_band_trace_count_by_pair=zero_traces[pi].tolist()))
            budget.check('PARTITION_%d_MATCHING' % pi)
            for ci, concept in enumerate(concepts):
                if ci % 64 == 0:
                    budget.check()
                rows = groups[concept]
                center_fold = 1 - concept_folds[ci]
                for variant_i, features in enumerate((raw_features, norm_features)):
                    full = [features[pair_i, rows].astype(np.float64) - centers[variant_i, pair_i, center_fold]
                            for pair_i in range(2)]
                    for scope_i, columns in enumerate((slice(None), DIAGONAL_INDICES)):
                        left, zero_left = unit_rows(full[0][:, columns])
                        right, zero_right = unit_rows(full[1][:, columns])
                        similarity = left @ right.T
                        assert np.isfinite(similarity).all() and np.max(np.abs(similarity)) <= 1 + 1e-10
                        for direction, matrix in enumerate((similarity, similarity.T)):
                            correct = matching_metrics(matrix, np.arange(10))
                            control = matching_metrics(matrix, (np.arange(10) + 1) % 10)
                            results[ci, variant_i, scope_i, pi, direction, 0] = correct.mean(axis=0)
                            results[ci, variant_i, scope_i, pi, direction, 1] = control.mean(axis=0)
                            results[ci, variant_i, scope_i, pi, direction, 2] = (correct - control).mean(axis=0)
                            stimulus_sums[rows, variant_i, scope_i, 0] += correct
                            stimulus_sums[rows, variant_i, scope_i, 1] += control
                            stimulus_sums[rows, variant_i, scope_i, 2] += correct - control
                            stimulus_counts[rows, variant_i, scope_i] += 1
                        zero_features[variant_i, scope_i, pi, 0] += zero_left
                        zero_features[variant_i, scope_i, pi, 1] += zero_right
            del raw_features, norm_features, sums, centers, full, left, right, similarity
            budget.check('PARTITION_%d_COMPLETE' % pi)
            budget.write('progress_partition_%d.json' % pi, dict(partitions_complete=pi + 1,
                         elapsed_seconds=time.monotonic() - PROCESS_START,
                         peak_extra_RSS_bytes=budget.peak_rss - budget.base_rss))
        assert np.isfinite(results).all()
        budget.check('FIXED_CONCEPT_BOOTSTRAP')
        bootstrap_indices = np.random.default_rng(1729).integers(0, 1323, size=(1000, 1323))
        all_directions = bootstrap_summary(results, bootstrap_indices)
        budget.check()
        # Partition/direction means are additional fixed summaries, never a
        # choice of the best partition, scope, normalization or frequency band.
        assert (stimulus_counts == 6).all()
        stimulus_average = stimulus_sums / 6
        # The six views share the same four observations. Pool them within each
        # stimulus first; they are never six independent replication units.
        average_directions = np.stack([stimulus_average[groups[concept]].mean(axis=0)
                                       for concept in concepts])
        assert np.allclose(average_directions, results.mean(axis=(3, 4)), atol=1e-13, rtol=1e-13)
        overall = bootstrap_summary(average_directions, bootstrap_indices)
        full_minus_diag = bootstrap_summary(average_directions[:, :, 0] - average_directions[:, :, 1], bootstrap_indices)
        budget.check('WRITE_ALL_CONCEPT_RESULTS')
        per_concept_path = args.output / 'per_concept.jsonl'
        assert not per_concept_path.exists()
        with per_concept_path.open('x', encoding='utf-8') as stream:
            for ci, concept in enumerate(concepts):
                value = dict(concept_id=concept, fold=concept_folds[ci], stimulus_rows=groups[concept],
                             metrics=results[ci].tolist())
                stream.write(json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n')
                if ci % 128 == 0:
                    budget.check()
            stream.flush()
            os.fsync(stream.fileno())
        assert budget.output_size() < OUTPUT_CAP
        stimulus_path = args.output / 'per_stimulus_all6_average.npy'
        with stimulus_path.open('xb') as stream:
            np.save(stream, stimulus_average, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        assert budget.output_size() < OUTPUT_CAP
        budget.checked(trials_ref)
        budget.checked(index_ref)
        budget.checked(contract_ref)
        budget.check('FINAL_SHA_AND_RESOURCE_CHECKS')
        summary = dict(
            status='T_ONLY_REPEAT_DIFFERENCE_SPECTRAL_STRUCTURE_CPU_COMPLETE',
            source=budget.ref(__file__), input_contract=contract_ref,
            inputs_actually_read=[trials_ref, index_ref], target_F_c_or_image_arrays_read=False,
            input_shapes=dict(trials=[13230, 4, 17, 250], features_per_pair=[13230, 1734]),
            subject='sub-01', T_stimuli=13230, T_concepts=1323, folds=fold_counts.tolist(),
            metadata_timebase=dict(sampling_rate_Hz=250, points=250, frequency_resolution_Hz=1,
                                   source='Existing locked250Hz array metadata, not a new raw-event reconstruction'),
            bands_Hz=BANDS, band_bin_counts=list(map(len, band_indices)),
            one_sided_frequency_weights='DC and Nyquist1; all interior bins2',
            band_weight_sums=[float(ONE_SIDED_WEIGHTS[bins].sum()) for bins in band_indices],
            Fourier_normalization='NumPy rfft default backward; no taper, demeaning, band selection or fitted whitening',
            pair_difference='(x_r-x_s)/sqrt(2)', covariance='sum_f w_f D_f D_f.conj().T / sum_f w_f',
            Hermitian_vector='diag17; sqrt(2)*real upper136; sqrt(2)*imag upper136; six bands concatenated; complex cross-spectrum including imaginary entries, not only real cospectrum',
            trace_normalization='Each band vector divided by its positive trace; zero trace gives zero vector',
            feature_views=dict(variants=VARIANTS, scopes=SCOPES, full_dim=1734, diagonal_dim=102),
            pairs=PAIRS, disjoint_partitions=PARTITIONS, directions=('left_to_right', 'right_to_left'),
            centering='Separate pair/variant mean from all stimuli in the opposite concept fold only; diagonal scope uses corresponding mean coordinates',
            gallery='All ten same-concept stimuli; no cross-concept labels used in fitting',
            cyclic_control='Correct gallery label is shifted by+1 in sorted compact-row order within each ten-stimulus concept',
            tie_policy='Top1 correct among k tied maxima gets1/k; 2AFC ties get0.5; centered zero-norm vectors get zero cosine; full denominator retained',
            saved_array_axes=('variant', 'scope', 'partition', 'direction', 'condition', 'metric'),
            conditions=CONDITIONS, metrics=METRICS,
            all_fixed_directions=all_directions,
            average_all3_partitions_and2_directions=dict(axes=('variant', 'scope', 'condition', 'metric'), **overall),
            primary_readout=dict(variant='raw_covariance', scope='full_Hermitian',
                                 correct_pairing_mean=average_directions[:, 0, 0, 0].mean(axis=0).tolist(),
                                 cyclic_wrong_pairing_mean=average_directions[:, 0, 0, 1].mean(axis=0).tolist(),
                                 correct_minus_control_mean=average_directions[:, 0, 0, 2].mean(axis=0).tolist(),
                                 interpretation='Prespecified primary diagnostic readout; review reproducibility only, no automatic model or image admission'),
            full_minus_diagonal=dict(axes=('variant', 'condition', 'metric'),
                                     claim_limit='Descriptive feature-scope difference; it does not independently establish cross-electrode coupling or a causal mechanism',
                                     **full_minus_diag),
            feature_center_statistics=feature_stats, zero_feature_rows=zero_features.tolist(),
            per_concept=budget.ref(per_concept_path), synthetic_checks=synthetic_ref,
            per_stimulus_all6_average=budget.ref(stimulus_path),
            within_stimulus_aggregation='Average all3 partitions x2 directions first, then average ten stimuli within concept, then1323concepts. Six views share four trials and are not independent replications.',
            bootstrap=dict(seed=1729, repetitions=1000, unit='all1323whole concepts',
                           shared_indices_for_all_views_and_contrasts=True,
                           confidence_intervals_descriptive=True, independent_confirmation=False),
            peak_extra_RSS_bytes=budget.peak_rss - budget.base_rss,
            peak_RSS_bytes=budget.peak_rss, maximum_extra_RSS_bytes=EXTRA_RSS_CAP,
            maximum_output_bytes=OUTPUT_CAP, maximum_seconds=args.max_seconds,
            thread_environment={name: os.environ[name] for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
            seconds=time.monotonic() - PROCESS_START, new_models=0, training_updates=0, new_training_updates=0,
            new_images=0, GPU_imported=False, GPU_initialized=False, CUDA_initialized=False,
            automatic_image_queue=False, ordinary_changed=False, sampler_changed=False,
            formal_D_resumed=False, image_gain_established=False, novelty_established=False,
            independent_session_metadata_available=False, iid_noise_established=False,
            biological_induced_activity_separated=False, information_beyond_c_established=False,
            limitations=[
                'Repeat differences measure repeat variation; ERP amplitude/latency jitter, instrumentation and session effects can enter them.',
                'Disjoint trial records do not establish independent sessions or iid errors; released preprocessing lineage remains limited.',
                'Reproducibility does not establish induced neural origin, EEG-only causal information, increment beyond c, or image-generation benefit.',
                'All four feature views and every partition/direction are reported; no winner is selected and no model/image job is automatically admitted.'
            ])
        summary_ref = budget.write('summary.json', summary)
        assert budget.output_size() < OUTPUT_CAP
        budget.check('COMPLETE')
        print(json.dumps(dict(status=summary['status'], summary=summary_ref,
                              seconds=time.monotonic() - PROCESS_START,
                              peak_extra_RSS_bytes=budget.peak_rss - budget.base_rss,
                              output_bytes=budget.output_size())))
    except BaseException as exc:
        failure = dict(status='FAILED_FINITE_TONLY_CPU_PROBE_NO_AUTOMATIC_RETRY', stage=budget.stage,
                       error=repr(exc), traceback=traceback.format_exc(), seconds=time.monotonic() - PROCESS_START,
                       peak_extra_RSS_bytes=budget.peak_rss - budget.base_rss,
                       no_automatic_retry=True, new_images=0, new_training_updates=0,
                       ordinary_changed=False, formal_D_resumed=False)
        path = args.output / 'failure.json'
        if not path.exists():
            path.write_text(json.dumps(failure, indent=2) + '\n', encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
