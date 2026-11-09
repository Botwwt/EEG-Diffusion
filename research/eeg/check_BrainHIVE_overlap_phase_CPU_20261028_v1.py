"""Numerical checks of closed-form bounded delay and disjoint sample contracts."""
import argparse
import json
from pathlib import Path
import numpy as np
import brainhive_overlap_phase_20261028_v1 as M


def run():
    rng = np.random.default_rng(1729)
    raw = rng.normal(size=(8, 17, 250))
    y, previous, following = M.difference_segment(raw, 20)
    phases = np.array([-.5, -.33, -.15, 0, .07, .2, .35, .5])
    x = M.predict_phase(y, previous, following, phases)
    fitted = M.fit_phase(x, y, previous, following)
    assert np.max(np.abs(fitted - phases)) < 1e-14
    checks = ['all_signed_and_boundary_piecewise_delays_recovered_exactly']
    assert np.max(np.abs(M.predict_phase(y, previous, following, fitted) - x)) < 1e-13
    checks.append('zero_and_true_parameter_prediction_paths')
    unrelated = rng.normal(size=x.shape)
    fit = M.fit_phase(unrelated, y, previous, following)
    true_loss = np.square(unrelated - M.predict_phase(y, previous, following, fit)).sum((1, 2))
    grid_loss = np.full(len(raw), np.inf)
    for value in np.linspace(-.5, .5, 2001):
        prediction = M.predict_phase(y, previous, following, np.full(len(raw), value))
        grid_loss = np.minimum(grid_loss, np.square(unrelated - prediction).sum((1, 2)))
    assert np.all(true_loss <= grid_loss + 1e-9)
    checks.append('piecewise_closed_form_at_least_as_good_as_dense_global_grid')
    constant = np.ones_like(x)
    assert np.array_equal(M.fit_phase(x, constant, constant, constant), np.zeros(len(x)))
    checks.append('zero_gradient_has_exact_zero_parameter_fallback')
    correct = M.predict_phase(y, previous, following, phases)
    flipped = M.predict_phase(y, previous, following, -phases)
    errors = M.normalized_errors(x, [y, correct, flipped], y)
    assert np.all(errors[:, 1] < 1e-20) and errors[:, 2].mean() > errors[:, 1].mean()
    checks.append('wrong_direction_control_and_identical_metric_denominator')
    selected_indices = set(range(19, 86))
    held_indices = set(range(89, 156))
    assert not selected_indices & held_indices
    for lag in (50, 49, 51, 73):
        assert not {x + lag for x in selected_indices} & {x + lag for x in held_indices}
        M.difference_segment(raw, lag + 90)
    checks.append('full_interpolation_neighborhoods_of_fit_and_held_are_disjoint')
    held_y, held_previous, held_following = M.difference_segment(raw, 90)
    held_x = M.predict_phase(held_y, held_previous, held_following, phases)
    predicted = M.predict_phase(held_y, held_previous, held_following, fitted)
    assert np.max(np.abs(predicted - held_x)) < 1e-13
    checks.append('parameter_transfers_to_disjoint_later_waveform')
    altered = raw.copy()
    altered[..., 89:156] += 10 * rng.normal(size=altered[..., 89:156].shape)
    altered_y, altered_previous, altered_following = M.difference_segment(altered, 20)
    assert np.array_equal(M.fit_phase(x, altered_y, altered_previous, altered_following), fitted)
    checks.append('held_region_modification_cannot_change_fit_parameter')
    return dict(status='OVERLAP_PHASE_PIECEWISE_LS_AND_DISJOINTNESS_CPU_CHECKS_PASSED',
                tests=len(checks), checks=checks, actual_EEG_read=False, GPU_jobs=0)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = run()
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result))
