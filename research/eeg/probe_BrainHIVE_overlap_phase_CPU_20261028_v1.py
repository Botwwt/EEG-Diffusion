"""D34: explain fixed D33 pair residuals with a constrained sampling operator."""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
from collections import defaultdict
import datetime
import hashlib
import json
from pathlib import Path
import signal
import socket
import time
import traceback
import numpy as np
import brainhive_overlap_phase_20261028_v1 as M

START = time.monotonic()
ARMS = ('unchanged', 'correct_phase', 'opposite_phase', 'other_repeat_phase', 'reversed_gradient_fit')


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked(item):
    path = Path(item['path'])
    assert path.is_file() and sha(path) == item['sha256'], str(path)
    if 'bytes' in item:
        assert path.stat().st_size == item['bytes']
    return path


def ref(path):
    return dict(path=str(path), bytes=Path(path).stat().st_size, sha256=sha(path))


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def rss():
    return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:'))) * 1024


def run(spec_path):
    spec = read(spec_path)
    assert spec['status'] == 'D34_FIXED_PAIR_PHASE_EXPLANATION_SPEC_LOCKED'
    assert spec['lags'] == [50, 49, 51, 73] and spec['phase_bound_samples'] == .5
    assert spec['fit_start'] == 20 and spec['held_start'] == 90 and spec['samples_per_segment'] == 64
    assert spec['no_automatic_model_or_image_admission']
    output = Path(spec['output'])
    assert not output.exists()
    output.mkdir(parents=True)
    base = rss()
    stage = 'VERIFY_INPUTS'
    completed = []

    def budget():
        assert time.monotonic() - START < 850
        assert rss() - base < 10 << 30
        assert sum(p.stat().st_size for p in output.iterdir() if p.is_file()) < 128 << 20

    try:
        for item in spec['source_pins']:
            checked(item)
        primary = read(checked(spec['upstream_D33_completion']))
        assert primary['registered_strict_overlap_support'] is False
        assert read(checked(spec['upstream_D33_spec']))['status'] == 'D33_T_ONLY_TEMPORAL_OVERLAP_SPEC_LOCKED'
        assert read(checked(spec['authorization']))['formal_old_D_training_resumed'] is False
        index = [json.loads(x) for x in checked(spec['T_index']).read_text().splitlines()]
        split = read(checked(spec['split']))
        assert len(index) == 13230 and [x['array_row'] for x in index] == split['image_rows']['T']
        groups = defaultdict(list)
        for i, row in enumerate(index):
            assert row['row'] == i and row['partition'] == 'T' and row['subject_id'] == 'sub-01'
            groups[row['concept_id']].append(i)
        assert len(groups) == 1323 and all(len(x) == 10 for x in groups.values())
        trial_path = checked(spec['T_trials'])
        raw = np.load(trial_path, mmap_mode='r', allow_pickle=False)
        assert raw.shape == (13230, 4, 17, 250) and raw.dtype == np.float32
        raw = raw.reshape(52920, 17, 250)
        n = len(raw)
        phases = np.empty((n, 4, 2), np.float64)
        errors = np.empty((n, 4, len(ARMS)), np.float64)
        other = np.arange(n).reshape(-1, 4)
        other = np.roll(other, -1, axis=1).reshape(-1)
        assert np.array_equal(other // 4, np.arange(n) // 4)
        primary_by_lag = {x['lag']: x for x in primary['results']}
        for arm_lag, lag in enumerate(spec['lags']):
            stage = 'LAG_' + str(lag)
            matches = np.load(checked(primary_by_lag[lag]['matches']), allow_pickle=False)
            target = matches['target'][:, 0]
            assert target.shape == (n,) and np.all(target != np.arange(n))
            for lo in range(0, n, 128):
                hi = min(n, lo + 128)
                source_raw = np.asarray(raw[lo:hi])
                target_raw = np.asarray(raw[target[lo:hi]])
                x = M.difference_segment(source_raw, lag + 20)[0]
                y, previous, following = M.difference_segment(target_raw, 20)
                phases[lo:hi, arm_lag, 0] = M.fit_phase(x, y, previous, following)
                phases[lo:hi, arm_lag, 1] = M.fit_phase(x, y, previous, following, reversed_gradient=True)
                budget()
            for lo in range(0, n, 128):
                hi = min(n, lo + 128)
                x = M.difference_segment(np.asarray(raw[lo:hi]), lag + 90)[0]
                y, previous, following = M.difference_segment(np.asarray(raw[target[lo:hi]]), 90)
                delta = phases[lo:hi, arm_lag, 0]
                predictions = [y, M.predict_phase(y, previous, following, delta),
                               M.predict_phase(y, previous, following, -delta),
                               M.predict_phase(y, previous, following, phases[other[lo:hi], arm_lag, 0]),
                               M.predict_phase(y, previous, following, phases[lo:hi, arm_lag, 1], reversed_gradient=True)]
                errors[lo:hi, arm_lag] = M.normalized_errors(x, predictions, y)
                budget()
            q = errors[:, arm_lag].mean(0)
            record = dict(lag=lag, all_trial_pairs=n, mean_normalized_squared_error=dict(zip(ARMS, q.tolist())),
                          relative_mean_error_reduction=float((q[0] - q[1]) / q[0]),
                          median_phase_samples=float(np.median(phases[:, arm_lag, 0])),
                          phase_quantiles=np.quantile(phases[:, arm_lag, 0], [0, .01, .5, .99, 1]).tolist(),
                          phase_bound_saturation_fraction=float((np.abs(phases[:, arm_lag, 0]) >= .5 - 1e-12).mean()))
            completed.append(record)
            write(output / f'lag{lag}_summary.json', record)
            print(json.dumps(record), flush=True)
        np.save(output / 'all52920x4x2_phase_fits.npy', phases, allow_pickle=False)
        np.save(output / 'all52920x4x5_held_errors.npy', errors, allow_pickle=False)
        concept = np.stack([errors.reshape(13230, 4, 4, len(ARMS))[positions].mean((0, 1)) for positions in groups.values()])
        assert concept.shape == (1323, 4, 5)
        np.save(output / 'all1323_concept_errors.npy', concept, allow_pickle=False)
        indices = np.random.default_rng(1729).integers(0, 1323, size=(1000, 1323))
        draws = np.concatenate([concept[indices[lo:lo + 20]].mean(1) for lo in range(0, len(indices), 20)])
        point = concept.mean(0)
        reductions = (point[:, 0] - point[:, 1]) / point[:, 0]
        draw_reductions = (draws[:, :, 0] - draws[:, :, 1]) / draws[:, :, 0]
        control_benefits = point[0, [2, 3, 4]] - point[0, 1]
        draw_controls = draws[:, 0, [2, 3, 4]] - draws[:, 0, 1, None]
        lag_specific = reductions[0] - reductions[1:]
        draw_lag_specific = draw_reductions[:, 0, None] - draw_reductions[:, 1:]
        low, high = np.quantile(draw_reductions, [.025, .975], axis=0)
        control_low, control_high = np.quantile(draw_controls, [.025, .975], axis=0)
        lag_low, lag_high = np.quantile(draw_lag_specific, [.025, .975], axis=0)
        support = bool(reductions[0] >= .10 and low[0] > 0 and np.all(control_low > 0)
                       and np.all(lag_low > 0) and completed[0]['phase_bound_saturation_fraction'] < .25)
        assert sha(trial_path) == spec['T_trials']['sha256']
        value = dict(status='D34_ALL_T_FIXED_PAIR_SAMPLING_PHASE_EXPLANATION_COMPLETE',
                     UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), host=socket.gethostname(),
                     spec=ref(spec_path), arms=ARMS, lags=spec['lags'], results=completed,
                     relative_reductions=reductions.tolist(), relative_reductions_descriptive_95_lower=low.tolist(),
                     relative_reductions_descriptive_95_upper=high.tolist(),
                     lag50_error_benefit_vs_controls=control_benefits.tolist(), control_order=list(ARMS[2:]),
                     lag50_error_benefit_vs_controls_descriptive_95_lower=control_low.tolist(),
                     lag50_error_benefit_vs_controls_descriptive_95_upper=control_high.tolist(),
                     lag50_relative_gain_minus_wrong_lags=lag_specific.tolist(),
                     lag50_relative_gain_minus_wrong_lags_descriptive_95_lower=lag_low.tolist(),
                     lag50_relative_gain_minus_wrong_lags_descriptive_95_upper=lag_high.tolist(),
                     registered_linear_interpolation_phase_explanation_supported=support,
                     decision='PHASE_OPERATOR_HAS_CROSS_TIME_SUPPORT_NOT_EVENT_OR_IMAGE_ADMISSION' if support
                     else 'STOP_THIS_BOUNDED_PHASE_EXPLANATION_NO_THRESHOLD_OR_INTERPOLATOR_SCAN',
                     all_phases=ref(output / 'all52920x4x2_phase_fits.npy'),
                     all_held_errors=ref(output / 'all52920x4x5_held_errors.npy'),
                     concept_errors=ref(output / 'all1323_concept_errors.npy'),
                     analytical_one_parameter_local_fits=52920 * 4 * 2,
                     old_D33_gate_or_matches_changed=False, target_images_or_F_read=False,
                     trained_neural_models=0, SGD_updates=0, new_GPU_jobs=0, new_images=0,
                     source_event_identity_established=False, response_demixing_identified=False,
                     no_automatic_model_or_image_admission=True,
                     uncertainty_limit='T is already exposed; concept bootstrap descriptive, unknown acquisition-chain dependence not modeled',
                     seconds=time.monotonic() - START, peak_extra_CPU_bytes=rss() - base)
        budget()
        write(output / 'completion.json', value)
        print(json.dumps(value), flush=True)
    except BaseException as exc:
        write(output / ('failure_' + str(time.time_ns()) + '.json'), dict(
            status='D34_FAILED_INCOMPLETE', stage=stage, completed_lags=[x['lag'] for x in completed],
            error=repr(exc), traceback=traceback.format_exc(), seconds=time.monotonic() - START))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(TimeoutError('Finite CPU deadline')))
    run(args.spec)
