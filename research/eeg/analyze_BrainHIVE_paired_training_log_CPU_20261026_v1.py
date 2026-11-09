"""Read-only, standard-library audit of CLOSED O/A/D/U training logs.

Only loss_mean is an all-update cumulative quantity. Every other loss/energy/
gradient diagnostic is the single batch observed at each registered log event.
No event-spacing weighting, fitted trend, GPU import, or image score access.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

PRIMARY_QUERIES = 132300
PASSES = 3
BATCH = 16
UPDATES_PER_PASS = (PRIMARY_QUERIES + BATCH - 1) // BATCH
TOTAL_UPDATES = PASSES * UPDATES_PER_PASS
TOTAL_EXPOSURES = PASSES * PRIMARY_QUERIES
WINDOW = 32
ARMS = ('O', 'A', 'D', 'U')
DIAGNOSTICS = ('valid_fraction', 'shuffled_valid_fraction', 'teacher_separation',
               'shuffled_teacher_separation', 'common_component',
               'contrast_component', 'student_difference_energy')
EVENT_UPDATES = [u for u in range(1, TOTAL_UPDATES + 1)
                 if u % 64 == 0 or u % UPDATES_PER_PASS == 0]
MAXIMUM_INPUT_BYTES = 16 << 20


def sha(content):
    return hashlib.sha256(content).hexdigest()


def read_bytes(path):
    value = Path(path).read_bytes()
    assert len(value) <= MAXIMUM_INPUT_BYTES, str(path)
    return value


def number(value, *, nonnegative=False):
    assert type(value) in (int, float) and math.isfinite(value), value
    if nonnegative:
        assert value >= 0, value
    return float(value)


def close(a, b):
    return math.isclose(a, b, rel_tol=5e-5, abs_tol=2e-6)


def quantile(values, p):
    ordered = sorted(values)
    location = (len(ordered) - 1) * p
    lower = int(location)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = location - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def describe(values):
    values = [number(v) for v in values if v is not None]
    if not values:
        return dict(defined_events=0)
    return dict(defined_events=len(values), mean=statistics.fmean(values),
                median=statistics.median(values), population_sd=statistics.pstdev(values),
                minimum=min(values), q10=quantile(values, .1), q90=quantile(values, .9),
                maximum=max(values))


def event_metrics(row):
    result = {k: row[k] for k in ('epsilon_loss', 'token_loss', 'auxiliary', 'gradient_norm')}
    result.update({'terms.' + k: row['terms'][k] for k in ('A', 'D', 'U')})
    result.update({'diagnostics.' + k: row['diagnostics'][k] for k in DIAGNOSTICS})
    absolute = row['terms']['A']
    teacher = row['diagnostics']['teacher_separation']
    result['derived.common_share_of_absolute_objective'] = (
        row['diagnostics']['common_component'] / absolute if absolute > 0 else None)
    result['derived.contrast_share_of_absolute_objective'] = (
        row['terms']['D'] / absolute if absolute > 0 else None)
    result['derived.ratio_of_student_to_teacher_layer_equal_batch_mean_energy'] = (
        row['diagnostics']['student_difference_energy'] / teacher if teacher > 0 else None)
    result['derived.weighted_auxiliary_loss'] = .1 * row['auxiliary']
    result['derived.weighted_token_loss'] = .1 * row['token_loss']
    return result


def summary(rows):
    assert rows
    values = [event_metrics(row) for row in rows]
    metrics = {key: describe([v[key] for v in values]) for key in values[0]}
    # Batch size describes only the last batch of each event. It does not turn
    # the intervening unlogged updates into observed component diagnostics.
    sample_exposures = sum(12 if row['updates'] % UPDATES_PER_PASS == 0 else 16 for row in rows)
    return dict(event_count=len(rows), first_update=rows[0]['updates'],
                last_update=rows[-1]['updates'], observed_batch_exposures=sample_exposures,
                estimand='EQUAL_WEIGHT_MEAN_OF_RECORDED_LAST_BATCH_SNAPSHOTS_ONLY',
                total_gradient_norm_above_clip_threshold_events=sum(row['gradient_norm'] > 1 for row in rows),
                metrics=metrics)


def validate_rows(rows, arm):
    assert arm in ARMS
    assert len(rows) == len(EVENT_UPDATES), 'Closed logs need every registered event'
    assert [row['updates'] for row in rows] == EVENT_UPDATES, 'Missing, duplicate, reordered or unexpected update'
    previous_seconds = -1
    for row in rows:
        update = row['updates']
        epoch = (update - 1) // UPDATES_PER_PASS
        within = (update - 1) % UPDATES_PER_PASS + 1
        expected_seen = epoch * PRIMARY_QUERIES + min(within * BATCH, PRIMARY_QUERIES)
        assert row['status'] == 'PAIRED_EFFECT_FULL_T_TRAINING' and row['arm'] == arm
        assert row['expected_updates'] == TOTAL_UPDATES
        assert row['expected_primary_exposures'] == TOTAL_EXPOSURES
        assert row['cursor'] == row['seen_primary_queries'] == row['seen_paired_queries'] == expected_seen
        assert row['UNet_forwards'] == 2
        for key in ('loss_mean', 'last_loss', 'epsilon_loss', 'token_loss', 'auxiliary', 'gradient_norm', 'seconds'):
            number(row[key], nonnegative=True)
        assert row['seconds'] >= previous_seconds
        previous_seconds = row['seconds']
        assert set(row['terms']) == {'A', 'D', 'U'}
        assert set(row['diagnostics']) == set(DIAGNOSTICS)
        for value in (*row['terms'].values(), *row['diagnostics'].values()):
            number(value, nonnegative=True)
        diag = row['diagnostics']
        assert 0 <= diag['valid_fraction'] <= 1
        assert 0 <= diag['shuffled_valid_fraction'] <= 1
        assert diag['valid_fraction'] == diag['shuffled_valid_fraction'], 'Same joint mask is required'
        assert close(row['terms']['A'], diag['common_component'] + row['terms']['D'])
        assert close(row['terms']['D'], diag['contrast_component'])
        assert close(row['auxiliary'], 0 if arm == 'O' else row['terms'][arm])
        assert close(row['last_loss'], row['epsilon_loss'] + .1 * row['token_loss'] + .1 * row['auxiliary'])


def analyze_rows(rows, arm):
    validate_rows(rows, arm)
    by_pass = {}
    previous_sum = 0.
    previous_seen = 0
    for epoch in range(PASSES):
        events = [row for row in rows if (row['updates'] - 1) // UPDATES_PER_PASS == epoch]
        terminal = events[-1]
        assert terminal['updates'] == (epoch + 1) * UPDATES_PER_PASS
        total_sum = terminal['loss_mean'] * terminal['seen_primary_queries']
        exact_pass_mean = (total_sum - previous_sum) / (terminal['seen_primary_queries'] - previous_seen)
        number(exact_pass_mean, nonnegative=True)
        by_pass[str(epoch + 1)] = dict(all_recorded_events=summary(events),
            first_fixed32_events=summary(events[:WINDOW]), last_fixed32_events=summary(events[-WINDOW:]),
            all_update_total_objective_mean_from_cumulative_sum=exact_pass_mean,
            primary_exposures=PRIMARY_QUERIES, paired_exposures=PRIMARY_QUERIES)
        previous_sum, previous_seen = total_sum, terminal['seen_primary_queries']
    whole = summary(rows)
    return dict(arm=arm, registered_event_count=len(EVENT_UPDATES),
        completed_updates=TOTAL_UPDATES, all_update_primary_exposures=TOTAL_EXPOSURES,
        all_update_paired_exposures=TOTAL_EXPOSURES,
        exact_all_update_total_objective_mean=rows[-1]['loss_mean'],
        snapshot_exposure_fraction=whole['observed_batch_exposures'] / TOTAL_EXPOSURES,
        all_recorded_events=whole, first_fixed32_events=summary(rows[:WINDOW]),
        last_fixed32_events=summary(rows[-WINDOW:]), per_pass=by_pass,
        contract_checks=dict(complete_event_sequence=True, exact_exposure_schedule=True,
            A_equals_common_plus_D=True, shared_joint_mask_fraction=True,
            active_auxiliary_matches_arm=True, logged_loss_recomposes=True,
            finite_nonnegative_metrics=True, elapsed_nondecreasing=True))


def load_closed(arm, directory):
    directory = Path(directory).resolve()
    payloads = {name: read_bytes(directory / name) for name in
                ('training.jsonl', 'completion.json', 'input_provenance.json', 'calibration.json')}
    done, provenance = [json.loads(payloads[name]) for name in ('completion.json', 'input_provenance.json')]
    assert done['status'] == 'PAIRED_EFFECT_FIXED_FINAL_FULL_T_COMPLETE'
    assert done['arm'] == provenance['arm'] == arm
    assert done['updates'] == TOTAL_UPDATES and done['passes'] == PASSES
    assert done['primary_exposures'] == done['paired_exposures'] == TOTAL_EXPOSURES
    assert done['parameters'] == provenance['head_parameters'] == 525568
    assert provenance['primary_query_exposures'] == provenance['paired_query_exposures'] == TOTAL_EXPOSURES
    assert provenance['training_seed'] == 1729 and provenance['input_role'] == 'OOF_T'
    assert done['study_core_sha256'] == provenance['study_core_sha256']
    assert done['provenance']['sha256'] == sha(payloads['input_provenance.json'])
    assert done['calibration']['sha256'] == sha(payloads['calibration.json'])
    assert payloads['training.jsonl'].endswith(b'\n'), 'Preserve incomplete log tails rather than silently skipping'
    rows = [json.loads(line) for line in payloads['training.jsonl'].splitlines()]
    result = analyze_rows(rows, arm)
    result.update(source_directory=str(directory), study_core_sha256=done['study_core_sha256'],
        input_sha256={name: sha(raw) for name, raw in payloads.items()},
        calibration=json.loads(payloads['calibration.json']),
        checkpoint_reference_from_completion=done['checkpoint'],
        checkpoint_bytes_not_loaded_or_reverified=True)
    return rows, result


def compare_logged_teacher_schedules(arms):
    result = {}
    names = list(arms)
    for left in names:
        for right in names[names.index(left) + 1:]:
            a, b = arms[left][0], arms[right][0]
            differences = {key: max(abs(x['diagnostics'][key] - y['diagnostics'][key]) for x, y in zip(a, b))
                           for key in ('teacher_separation', 'shuffled_teacher_separation',
                                       'valid_fraction', 'shuffled_valid_fraction')}
            result[left + '_vs_' + right] = dict(paired_events=len(a),
                maximum_absolute_difference=differences, exactly_equal=all(v == 0 for v in differences.values()),
                scope='TEACHER_AND_MASK_SCHEDULE_SANITY_ONLY_NOT_METHOD_EFFECT')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', action='append', required=True, metavar='ARM=DIRECTORY')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve prior analysis; use a new evidence path'
    arms = {}
    for spec in args.input:
        arm, directory = spec.split('=', 1)
        assert arm in ARMS and arm not in arms
        arms[arm] = load_closed(arm, directory)
    assert len({value[1]['study_core_sha256'] for value in arms.values()}) == 1
    result = dict(status='CLOSED_PAIRED_TRAINING_LOG_CPU_SNAPSHOT_ANALYSIS_COMPLETE',
        analyzer_sha256=sha(Path(__file__).read_bytes()), GPU_used=False,
        training_or_inference_started=False, protected_image_scores_read=False,
        predefined_window_events=WINDOW, arms={k: v[1] for k, v in arms.items()},
        cross_arm_teacher_schedule_checks=compare_logged_teacher_schedules(arms),
        interpretation_limits=[
            'Component losses, masks, energies and gradient norms describe only recorded last-batch snapshots, not all updates or queries.',
            'Only loss_mean and differences of its pass-end cumulative sums summarize all-update total objective.',
            'common/A is the share of the normalized absolute surrogate at the observed snapshots, not a share of causal image quality.',
            'Student/teacher energy is a ratio of layer-equal batch means, not mean per-layer normalized energy or recoverable EEG information.',
            'A total gradient norm does not identify per-objective gradients, conflicts or the active contribution of an auxiliary term.',
            'No per-query, per-layer, timestep, correction-radius, matched-versus-shuffled student, conditional covariance or held-out image outputs are present.',
            'These logs cannot establish information collapse, prevent regression shrinkage, prove stimulus-specific recovery or demonstrate image gains.',
        ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], arms=list(arms),
        event_counts={k: v[1]['registered_event_count'] for k, v in arms.items()},
        output=str(args.output), output_sha256=sha(args.output.read_bytes()))))


if __name__ == '__main__':
    main()
