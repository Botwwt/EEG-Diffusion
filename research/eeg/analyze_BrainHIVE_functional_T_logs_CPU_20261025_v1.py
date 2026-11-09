"""Four-arm training-log analysis only; no D/V/test arrays, images, or scores.

Logs sample optimization every 64 updates plus pass ends. They do not contain
per-layer/per-time signals or separate loss gradients. This tool cannot infer
those missing quantities, nor turn smaller training loss into image benefit.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import time

from diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v1 import (
    ARMS, ROOT, decision, inspect_finals, read, sha, write,
)

SCHEMA = 'BrainHIVE_functional_T_logs_CPU_20261025_v1'
COMPLETE = 'COMPLETE_FOUR_ARM_TRAINING_LOG_ANALYSIS_OPTIMIZATION_ONLY'
REQUIRED_SOURCES = (Path(__file__).name, 'diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v1.py',
                    'brainhive_functional_training_contract_v2.py')
EXPECTED_UPDATES = sorted(set(range(64, 24808, 64)) | {8269, 16538, 24807})
METRICS = ('last_loss', 'epsilon_loss', 'token_loss', 'functional_auxiliary',
           'gradient_norm', 'correction_RMS')


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def describe(values):
    assert values and all(finite(value) for value in values)
    ordered = sorted(values)
    return dict(count=len(values), mean=statistics.fmean(values), minimum=ordered[0], maximum=ordered[-1],
                median=statistics.median(values), fraction_at_least_one=statistics.fmean(value >= 1 for value in values))


def analyze_records(records, arm):
    assert arm in ARMS and records
    latest, duplicate_count, rollback_count = {}, 0, 0
    previous = -1
    for row in records:
        assert row['stage'] == 'BRAINHIVE_FUNCTIONAL_ALIGNMENT_V2_REAL_OOF_TRAINING'
        assert row['arm'] == arm and row['expected_updates'] == 24807 and row['expected_queries'] == 396900
        update = row['updates']
        assert type(update) is int and update in EXPECTED_UPDATES
        assert row['pass_number'] in (1, 2, 3) and row['UNet_forwards'] == 2 and row['mean_gate'] == 1
        assert all(finite(row[key]) for key in METRICS + ('cumulative_mean_loss', 'seconds'))
        assert all(row[key] >= 0 for key in METRICS + ('cumulative_mean_loss', 'seconds'))
        assert 0 < row['seen_queries'] <= 396900
        assert all(finite(value) for value in row['functional_diagnostics'].values())
        if update in latest:
            duplicate_count += 1
        if update < previous:
            rollback_count += 1
        latest[update] = row
        previous = update
    assert set(latest) == set(EXPECTED_UPDATES), ('Missing registered logged update points', arm,
        sorted(set(EXPECTED_UPDATES) - set(latest)))
    ordered = [latest[update] for update in EXPECTED_UPDATES]
    assert ordered[-1]['updates'] == 24807 and ordered[-1]['seen_queries'] == 396900
    # Latest record for each step preserves resumed updates. Duplicates and
    # rollbacks are reported explicitly, never silently treated as new queries.
    width = max(1, math.ceil(len(ordered) * .2))
    first, last = ordered[:width], ordered[-width:]
    metrics = {}
    for key in METRICS:
        early = describe([row[key] for row in first])
        late = describe([row[key] for row in last])
        metrics[key] = dict(all_logged=describe([row[key] for row in ordered]), first_20_percent=early,
                            last_20_percent=late, last_minus_first=late['mean'] - early['mean'])
    common_diagnostics = set.intersection(*(set(row['functional_diagnostics']) for row in ordered))
    auxiliary = {}
    for key in sorted(common_diagnostics):
        early = statistics.fmean(row['functional_diagnostics'][key] for row in first)
        late = statistics.fmean(row['functional_diagnostics'][key] for row in last)
        auxiliary[key] = dict(first_20_percent=early, last_20_percent=late, last_minus_first=late - early)
    return dict(arm=arm, raw_logged_records=len(records), unique_logged_updates=len(ordered),
        expected_logged_updates=len(EXPECTED_UPDATES), duplicate_updates=duplicate_count,
        rollback_boundaries=rollback_count, complete_registered_log_schedule=True,
        logged_update_count_is_not_loss_for_all_24807_updates=True,
        selection='Latest retained record per registered update; all duplicates and rollbacks counted.',
        metrics=metrics, functional_diagnostics=auxiliary,
        clipped_update_proxy_fraction=metrics['gradient_norm']['all_logged']['fraction_at_least_one'],
        pre_clip_gradient_norm_logged=True,
        correction_RMS_does_not_identify_radius_utilization=True,
        by_pass={str(number): {key: describe([row[key] for row in ordered if row['pass_number'] == number])
                              for key in METRICS} for number in (1, 2, 3)},
        final_logged_cumulative_mean_loss=ordered[-1]['cumulative_mean_loss'])


def validate_registration(document):
    assert document['schema'] == SCHEMA and document['status'] == 'REGISTERED_T_ONLY_FROZEN_DIAGNOSTIC'
    assert document['seed'] == 1729 and set(document['formal_recipes']) == set(ARMS)
    assert Path(document['output_directory']).is_absolute()
    assert set(REQUIRED_SOURCES) <= set(document['source_sha256'])
    for name in REQUIRED_SOURCES:
        assert sha(ROOT / name) == document['source_sha256'][name], ('source mismatch', name)
    for ref in document['formal_recipes'].values():
        assert sha(ref['path']) == ref['sha256']
    return document


def execute(document, registration_path):
    started = time.monotonic()
    finals = inspect_finals(document)
    output = Path(document['output_directory'])
    output.mkdir(parents=True, exist_ok=False)
    analyses, provenance = {}, {}
    try:
        for arm, final in finals.items():
            path = Path(final['recipe']['output_directory']) / 'training.jsonl'
            before = sha(path)
            assert path.stat().st_size <= 64 << 20
            with path.open(encoding='utf-8') as stream:
                records = [json.loads(line) for line in stream if line.strip()]
            analyses[arm] = analyze_records(records, arm)
            assert sha(path) == before
            provenance[arm] = dict(path=str(path), sha256=before,
                completion_sha256=final['completion_sha256'], checkpoint_sha256=final['checkpoint_sha256'])
        assert time.monotonic() - started < 300
        epsilon_deltas = {arm: analyses[arm]['metrics']['epsilon_loss']['last_minus_first'] for arm in ARMS}
        unstable = any(analysis['metrics']['last_loss']['last_20_percent']['maximum'] >
                       10 * max(analysis['metrics']['last_loss']['first_20_percent']['median'], 1e-12)
                       for analysis in analyses.values())
        optimization = decision('NOT_SUPPORTED' if unstable else 'SUPPORTED',
            dict(all_logged_values_finite=True, late_loss_greater_than_10x_early_median=unstable,
                 early_late_epsilon_deltas=epsilon_deltas),
            'Finite logged updates and no fixed extreme-growth trigger only; not proof of complete-step stability or image quality.')
        aux_deltas = {arm: analyses[arm]['metrics']['functional_auxiliary']['last_minus_first'] for arm in ('P', 'PN', 'S')}
        aux_result = ('SUPPORTED' if all(value < 0 for value in aux_deltas.values()) else
                      'NOT_SUPPORTED' if all(value >= 0 for value in aux_deltas.values()) else 'UNKNOWN')
        summary = dict(status=COMPLETE, arms=analyses,
            diagnostic_decisions=dict(optimization_stability=optimization,
                auxiliary_objective_optimization=decision(aux_result, aux_deltas,
                    'Whether each active auxiliary objective decreases between fixed first/last logged windows; objectives differ across arms.')),
            input_provenance=provenance, registration_sha256=sha(registration_path),
            fixed_windows='First and last ceil(20% of the 390 unique registered log events), no best epoch selection.',
            unavailable=['per-query losses', 'timestep-conditioned losses', 'per-layer functional responses',
                         'separate objective gradients', 'gradient cosines', 'radius utilization', 'image improvement'],
            scope='Training optimization description only; all rows are T-only training records.',
            GPU_allocated=False, no_optimizer_created=True, parameter_updates=0,
            protected_scores_read=False, generated_images=0, candidate_auto_started=False,
            elapsed_seconds=time.monotonic() - started)
        # Check all final files again after parsing the entire set of logs.
        for arm, final in finals.items():
            assert sha(final['checkpoint_path']) == final['checkpoint_sha256']
            assert sha(final['completion_path']) == final['completion_sha256']
        write(output / 'training_log_summary.json', summary)
        write(output / 'completion.json', dict(status=COMPLETE, summary_sha256=sha(output / 'training_log_summary.json'),
            parameter_updates=0, GPU_allocated=False, protected_scores_read=False))
        print(COMPLETE, flush=True)
    except BaseException as error:
        write(output / 'failure.json', dict(status='FAILED_OR_INCOMPLETE_TRAINING_LOG_ANALYSIS',
            error_type=type(error).__name__, error=str(error), completed_arms=list(analyses), expected_arms=list(ARMS),
            no_missing_arm_or_log_event_dropped=True))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    args = parser.parse_args()
    execute(validate_registration(read(args.registration)), args.registration)


if __name__ == '__main__':
    main()
