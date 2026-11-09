"""Independent NumPy audit, without importing the primary overlap module."""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load_checked(item):
    path = Path(item['path'])
    assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
    return path


def run(spec_path, completion_path):
    start = time.monotonic()
    spec = json.loads(spec_path.read_bytes())
    primary = json.loads(completion_path.read_bytes())
    assert sha(spec_path) == primary['spec']['sha256']
    assert spec['status'] == 'D33_T_ONLY_TEMPORAL_OVERLAP_SPEC_LOCKED'
    raw = np.load(load_checked(spec['T_trials']), mmap_mode='r', allow_pickle=False).reshape(52920, 17, 250)
    coords = np.random.default_rng(1729).choice(17 * 64, 32, replace=False)
    channels, times = coords // 64, coords % 64
    sample = np.linspace(0, len(raw) - 1, 32, dtype=np.int64)
    results = []
    for record in primary['results']:
        lag = record['lag']
        left = np.load(load_checked(record['source_fingerprints']), mmap_mode='r', allow_pickle=False)
        right = np.load(load_checked(record['target_fingerprints']), mmap_mode='r', allow_pickle=False)
        arrays = np.load(load_checked(record['matches']), allow_pickle=False)
        target, scores = arrays['target'], arrays['scores']
        inverse, stored_verified = arrays['inverse'], arrays['verified']
        assert target.shape == (52920, 2) and stored_verified.shape == (52920, 3)
        max_fingerprint_error = 0.
        metrics = np.empty((len(raw), 3), np.float64)
        for lo in range(0, len(raw), 128):
            hi = min(lo + 128, len(raw))
            x = np.asarray(raw[lo:hi], np.float64)
            for stored, shift in ((left, lag), (right, 0)):
                fp = x[:, channels, times + shift + 21] - x[:, channels, times + shift + 20]
                fp /= np.sqrt(np.square(fp).sum(1, keepdims=True))
                error = np.abs(fp.astype(np.float32) - stored[lo:hi]).max()
                max_fingerprint_error = max(max_fingerprint_error, float(error))
                assert error <= 1e-7
            # Calculate the disjoint later interval directly from raw cache.
            a = x[:, :, lag + 84:lag + 149]
            b = np.asarray(raw[target[lo:hi, 0], :, 84:149], np.float64)
            da = a[:, :, 1:] - a[:, :, :-1]
            db = b[:, :, 1:] - b[:, :, :-1]
            energy_a, energy_b = np.square(da).sum((1, 2)), np.square(db).sum((1, 2))
            relative = np.sqrt(np.square(da - db).sum((1, 2)) / np.maximum((energy_a + energy_b) / 2, 1e-60))
            cosine = (da * db).sum((1, 2)) / np.maximum(np.sqrt(energy_a * energy_b), 1e-30)
            a = a - np.sum(a, axis=-1, keepdims=True) / 65
            b = b - np.sum(b, axis=-1, keepdims=True) / 65
            raw_relative = np.sqrt(np.square(a - b).sum((1, 2)) / np.maximum((np.square(a).sum((1, 2)) + np.square(b).sum((1, 2))) / 2, 1e-60))
            metrics[lo:hi] = np.column_stack((relative, cosine, raw_relative))
        error = float(np.abs(metrics - stored_verified).max())
        assert error < 1e-12
        forward = np.asarray(left[sample]) @ right.T
        forward[np.arange(len(sample)), sample] = -np.inf
        best = forward.argmax(1)
        assert np.array_equal(best, target[sample, 0])
        first = forward[np.arange(len(sample)), best].copy()
        forward[np.arange(len(sample)), best] = -np.inf
        second = forward.argmax(1)
        assert np.array_equal(second, target[sample, 1])
        assert np.allclose(first, scores[sample, 0], atol=5e-7, rtol=0)
        reverse = np.asarray(left) @ np.asarray(right[sample]).T
        reverse[sample, np.arange(len(sample))] = -np.inf
        assert np.array_equal(reverse.argmax(0), inverse[sample])
        reciprocal = inverse[target[:, 0]] == np.arange(len(raw))
        keep = (reciprocal & (scores[:, 0] >= .99999) & ((scores[:, 0] - scores[:, 1]) >= .0001)
                & (metrics[:, 0] <= .0001) & (metrics[:, 2] <= .0001))
        assert np.array_equal(keep, arrays['accepted']) and np.array_equal(reciprocal, arrays['reciprocal'])
        assert int(keep.sum()) == record['accepted']
        for key, col in [('held_difference_relative_error_quantiles', 0), ('held_difference_cosine_quantiles', 1),
                         ('held_demeaned_raw_relative_error_quantiles', 2)]:
            assert np.allclose(np.quantile(metrics[:, col], [0, .01, .5, .99, 1]), record[key], atol=1e-12, rtol=0)
        # This execution found no edges, so graph equality is exact without any
        # inferred event or cycle reconstruction from hypothetical matches.
        assert record['accepted'] == 0 and record['graph']['accepted_edges'] == 0
        results.append(dict(lag=lag, full_trial_metrics_replayed=len(raw), full_fingerprints_replayed=len(raw) * 2,
                            all_candidate_target_ids_in_range=bool(((target >= 0) & (target < len(raw))).all()),
                            global_search_forward_and_reverse_independent_queries=32,
                            maximum_fingerprint_abs_error=max_fingerprint_error,
                            maximum_verification_abs_error=error, accepted=int(keep.sum())))
    assert all(x['accepted'] == 0 for x in results) and not primary['registered_strict_overlap_support']
    return dict(status='D33_FULL_FINGERPRINTS_ALL_MATCH_METRICS_AND32QUERY_GLOBAL_SEARCH_INDEPENDENT_CPU_VERIFIED',
                UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                source_sha256=sha(__file__), spec_sha256=sha(spec_path), primary_completion_sha256=sha(completion_path),
                results=results, registered_gate_recomputed=True, registered_strict_overlap_support=False,
                primary_stop_confirmed=True,
                entire_N_squared_search_independently_repeated=False,
                source_event_identity_established=False, new_model_fits=0, new_GPU_jobs=0, new_images=0,
                seconds=time.monotonic() - start)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--completion', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = run(args.spec, args.completion)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(result))
