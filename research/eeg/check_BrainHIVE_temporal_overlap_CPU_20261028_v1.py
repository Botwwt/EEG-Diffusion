"""Meaningful synthetic checks before any real-T fingerprint analysis."""
import argparse
import json
from pathlib import Path
import numpy as np
import brainhive_temporal_overlap_20261028_v1 as O


def run():
    rng = np.random.default_rng(11)
    continuous = rng.normal(size=(17, 1250)).astype(np.float32)
    trials = rng.normal(size=(27, 17, 250)).astype(np.float32)
    for i in range(20):
        trials[i] = continuous[:, i * 50:i * 50 + 250] + rng.normal(size=(17, 1)).astype(np.float32)
    checks = []
    l, r = O.fingerprints(trials, 50)
    ids, scores, inverse, reverse = O.search(l, r, block=7)
    brute = l @ r.T
    np.fill_diagonal(brute, -np.inf)
    assert np.array_equal(ids[:, 0], brute.argmax(1))
    assert np.array_equal(inverse, brute.argmax(0))
    assert np.allclose(scores[:, 0], brute[np.arange(len(trials)), ids[:, 0]], atol=2e-7)
    checks.append('blocked_all_pairs_matches_dense_forward_reverse')
    verify = O.verify(trials, 50, ids[:, 0])
    keep, _ = O.accepted(ids, scores, inverse, verify)
    assert np.array_equal(ids[:19, 0], np.arange(1, 20)) and keep.sum() == 19
    checks.append('exact_continuous_overlap_with_per_channel_baseline_recovered')
    assert O.graph(len(trials), ids[:, 0], keep)['maximum_path_nodes'] == 20
    checks.append('twenty_stimulus_path_graph_recovered')
    for lag in (49, 51, 73):
        l, r = O.fingerprints(trials, lag)
        ids, scores, inverse, _ = O.search(l, r)
        verified = O.verify(trials, lag, ids[:, 0])
        keep, _ = O.accepted(ids, scores, inverse, verified)
        assert not keep.any()
    checks.append('all_three_wrong_lags_rejected')
    altered = trials.copy()
    altered[1] *= 2
    verified = O.verify(altered, 50, np.r_[np.arange(1, len(trials)), 0])
    assert verified[0, 0] > .1
    checks.append('proportional_fingerprint_is_insufficient_without_raw_scale_check')
    altered = trials.copy()
    altered[1, :, 100:150] += rng.normal(size=(17, 50))
    l, r = O.fingerprints(altered, 50)
    ids, scores, inverse, _ = O.search(l, r)
    verified = O.verify(altered, 50, ids[:, 0])
    keep, _ = O.accepted(ids, scores, inverse, verified)
    assert ids[0, 0] == 1 and not keep[0]
    checks.append('held_temporal_samples_reject_selection_only_match')
    try:
        O.fingerprints(np.zeros((3, 17, 250), np.float32), 50)
        raise RuntimeError('constant input was admitted')
    except AssertionError:
        pass
    checks.append('zero_energy_rejected')
    cyc = O.graph(4, np.array([1, 0, 3, 2]), np.ones(4, bool))
    assert cyc['cyclic_source_nodes'] == 4
    checks.append('cyclic_match_cannot_be_called_a_recording_path')
    return dict(status='EEG_TEMPORAL_OVERLAP_SYNTHETIC_CPU_CHECKS_PASSED', tests=len(checks), checks=checks,
                real_EEG_read=False, GPU_jobs=0, models_fit=0)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    value = run()
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
    print(json.dumps(value))
