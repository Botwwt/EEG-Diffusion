"""One exact CPU continuation of a terminated saved-image spatial audit.

Only original timeout rows are inspected. Completed images, first artifacts,
thresholds, targets and all semantic metrics remain untouched. There is one
canonical new output directory and no automatic retry or further continuation.
"""
import argparse
import collections
import csv
import gzip
import itertools
import json
import math
import os
from pathlib import Path
import time

import audit_BrainHIVE_spatial_image_integrity_CPU_v1 as original


SCHEMA = 'BrainHIVE_spatial_integrity_CPU_finish_once_v1'
FLAGS = ('exact_spatial_constant', 'near_spatial_constant', 'legacy_global_blank_proxy')


def proc_snapshot(pid):
    """A live/reused PID fails closed; zombies have no running audit worker."""
    if os.name != 'posix' or not Path('/proc').is_dir():
        raise RuntimeError('Actual Linux /proc termination evidence is required')
    folder = Path('/proc') / str(pid)
    try:
        fields = (folder / 'stat').read_text().rsplit(') ', 1)[1].split()
        return dict(pid=pid, exists=True, state=fields[0], live=fields[0] not in ('Z', 'X'),
                    start_ticks=fields[19])
    except (FileNotFoundError, ProcessLookupError):
        return dict(pid=pid, exists=False, state=None, live=False, start_ticks=None)


def require_terminated(pid):
    assert isinstance(pid, int) and pid > 0
    snapshot = proc_snapshot(pid)
    assert snapshot['pid'] == pid and snapshot['live'] is False, 'Original audit PID is still live or reused'
    assert snapshot['exists'] is False or snapshot['state'] in ('Z', 'X')
    return snapshot


def checked_sha(path, expected):
    assert isinstance(expected, str) and len(expected) == 64
    actual = original.sha_file(path)
    assert actual == expected, 'Immutable input SHA mismatch: ' + str(path)
    return actual


def read_rows(manifest, table, manifest_sha):
    """Stream both immutable inputs in exact line order, retaining original IDs."""
    with Path(manifest).open(encoding='utf-8') as source, gzip.open(table, 'rt', newline='') as cached:
        reader = csv.DictReader(cached)
        assert reader.fieldnames == original.FIELDS
        for line_number, pair in enumerate(itertools.zip_longest(source, reader), 1):
            line, row = pair
            assert line is not None and row is not None, 'Manifest/table record counts differ'
            record = json.loads(line)
            assert set(row) == set(original.FIELDS)
            assert row['source_line_number'] == str(line_number)
            assert row['source_manifest_sha256'] == manifest_sha
            for key in ('query_id', 'subject', 'stimulus_id', 'generated_path'):
                assert isinstance(record[key], str) and row[key] == record[key], (line_number, key)
            assert Path(record['generated_path']).is_absolute()
            assert row['manifest_generated_sha256'] == record['generated_sha256']
            assert len(record['generated_sha256']) == 64
            assert all(c in '0123456789abcdef' for c in record['generated_sha256'])
            for key in ('hash_verified', 'decoded_512_RGB'):
                assert row[key] in ('True', 'False')
            if row['actual_file_sha256']:
                assert len(row['actual_file_sha256']) == 64
                assert all(c in '0123456789abcdef' for c in row['actual_file_sha256'])
            if row['hash_verified'] == 'True':
                assert row['actual_file_sha256'] == record['generated_sha256']
            if row['decoded_512_RGB'] == 'True':
                assert row['width'] == row['height'] == '512' and row['original_mode'] == 'RGB'
                assert len(row['decoded_RGB_pixel_sha256']) == 64
                for key in FLAGS:
                    assert row[key] in ('True', 'False')
                for key in original.FIELDS[16:]:
                    if key not in FLAGS:
                        assert math.isfinite(float(row[key])), (line_number, key)
            else:
                assert not any(row[key] for key in FLAGS)
            yield record, row


def empty_counts():
    return dict(records=0, decoded=0, trusted=0, failures=collections.Counter(),
                flags=collections.Counter(), trusted_flags=collections.Counter(), pixels={})


def add_counts(counts, row):
    counts['records'] += 1
    issues = [] if row['status'] == 'OK' else row['status'].split('|')
    assert all(issues) or row['status'] == 'OK'
    counts['failures'].update(issues)
    decoded = row['decoded_512_RGB'] in (True, 'True')
    trusted = decoded and row['hash_verified'] in (True, 'True') and not issues
    counts['decoded'] += int(decoded)
    counts['trusted'] += int(trusted)
    if decoded:
        for name in FLAGS:
            flagged = row[name] in (True, 'True')
            counts['flags'][name] += int(flagged)
            counts['trusted_flags'][name] += int(flagged and trusted)
        group = counts['pixels'].setdefault(row['decoded_RGB_pixel_sha256'], [0, set()])
        group[0] += 1
        group[1].add(row['actual_file_sha256'])


def preflight(launch_receipt, launch_sha256, first_summary_sha256, first_table_sha256, expected_count):
    pins = {str(Path(launch_receipt)): checked_sha(launch_receipt, launch_sha256)}
    launch = json.loads(Path(launch_receipt).read_text())
    assert launch['status'] == 'ACTUAL_COMPLETE_RISK_SPATIAL_CPU_AUDIT_STARTED'
    assert launch['CPU_only'] is True and launch['new_training'] is False
    assert launch['new_images'] is False and launch['new_semantic_scores'] is False
    assert launch['maximum_seconds'] == original.MAX_SECONDS
    assert launch['maximum_output_bytes'] == original.MAX_OUTPUT_BYTES
    assert launch['expected_queries'] == expected_count
    terminated = require_terminated(launch['pid'])
    first = Path(launch['output_directory'])
    summary_path, table_path = first / 'summary.json', first / 'per_query.csv.gz'
    pins[str(summary_path)] = checked_sha(summary_path, first_summary_sha256)
    pins[str(table_path)] = checked_sha(table_path, first_table_sha256)
    summary = json.loads(summary_path.read_text())
    assert summary['schema'] == original.SCHEMA and 'continuation_count' not in summary
    assert summary['expected_queries'] == summary['full_manifest_denominator'] == expected_count
    assert summary['per_query_rows_written'] == summary['visited_records'] == expected_count
    assert summary['per_query_rows_unwritten'] == 0 and summary['source_errors'] == []
    assert summary['budget']['output_budget_exhausted'] is False
    assert summary['budget']['time_budget_exhausted'] is True
    assert summary['manifest_count_matches_expected'] is True
    source = summary['source']
    assert source['manifest_path'] == launch['manifest_path']
    assert source['manifest_sha256'] == launch['manifest_sha256']
    pins[launch['manifest_path']] = checked_sha(launch['manifest_path'], launch['manifest_sha256'])
    argv = launch['argv']
    for flag, value in (('--manifest', launch['manifest_path']), ('--output', str(first)),
                        ('--expected-count', str(expected_count)), ('--registration', launch['registration_path']),
                        ('--registration-sha256', launch['registration_sha256'])):
        assert argv.count(flag) == 1 and argv[argv.index(flag) + 1] == value
    assert source['tool_path'] in argv
    assert Path(source['tool_path']).resolve() == Path(original.__file__).resolve()
    for name, digest in launch['source_sha256'].items():
        assert Path(name).name == name
        path = Path(original.__file__).parent / name
        pins[str(path)] = checked_sha(path, digest)
    assert source['tool_sha256'] == launch['source_sha256'][Path(original.__file__).name]
    registration = original.validate_registration(launch['registration_path'], launch['registration_sha256'])
    assert source['registration'] == registration
    pins[launch['registration_path']] = launch['registration_sha256']
    if 'CPU_contract_path' in launch or 'CPU_contract_sha256' in launch:
        pins[launch['CPU_contract_path']] = checked_sha(launch['CPU_contract_path'], launch['CPU_contract_sha256'])
    counts = empty_counts()
    queries, paths, pending = set(), set(), 0
    for record, row in read_rows(launch['manifest_path'], table_path, launch['manifest_sha256']):
        assert record['query_id'] not in queries and record['generated_path'] not in paths
        queries.add(record['query_id'])
        paths.add(record['generated_path'])
        add_counts(counts, row)
        pending += int(row['status'] == 'not_processed_time_limit' and row['decoded_512_RGB'] == 'False')
    assert counts['records'] == expected_count and pending > 0
    assert counts['decoded'] == summary['decoded_512_RGB']
    assert counts['trusted'] == summary['hash_verified_decoded_without_other_failures']
    assert dict(counts['failures']) == summary['failures']
    for name in FLAGS:
        assert counts['flags'][name] == summary['decoded_flag_counts'][name]
        assert counts['trusted_flags'][name] == summary['hash_verified_flag_counts'][name]
    assert summary['spatial_flags_unknown'] == expected_count - counts['decoded']
    return launch, summary, table_path, pins, pending, terminated


def finish(launch_receipt, launch_sha256, first_summary_sha256, first_table_sha256, expected_count=16600):
    start = time.monotonic()
    for name in original.THREAD_VARIABLES:
        os.environ[name] = '1'
    memory = original.memory_preflight()
    launch, first_summary, first_table, pins, pending, terminated = preflight(
        launch_receipt, launch_sha256, first_summary_sha256, first_table_sha256, expected_count)
    output = Path(launch['output_directory']).with_name(Path(launch['output_directory']).name + '_finish_once_v1')
    assert not output.exists(), 'One canonical continuation already exists; no retry or third segment'
    require_terminated(launch['pid'])
    for path, digest in pins.items():
        checked_sha(path, digest)
    import numpy as np
    from PIL import Image
    import PIL
    for key, actual in (('numpy', np.__version__), ('Pillow', PIL.__version__)):
        if key in launch.get('runtime_versions', {}):
            assert launch['runtime_versions'][key] == actual, 'CPU measurement runtime changed'
    Image.MAX_IMAGE_PIXELS = 512 * 512
    output.mkdir(exist_ok=False)
    budget = original.OutputBudget(output, original.MAX_OUTPUT_BYTES)
    table = original.ChunkedCSV(budget, 'per_query.csv.gz', original.FIELDS)
    deadline = start + original.MAX_SECONDS
    counts = empty_counts()
    attempted = completed = 0
    time_exhausted = output_exhausted = False
    source_errors = []

    def progress(stage):
        value = dict(stage=stage, utc=original.utc_now(), merged_records=counts['records'],
            manifest_denominator=expected_count, original_pending=pending,
            continuation_attempted=attempted, continuation_decoded=completed,
            decoded_512_RGB=counts['decoded'], hash_verified_decoded=counts['trusted'],
            seconds=time.monotonic() - start, time_budget_exhausted=time_exhausted,
            output_budget_exhausted=output_exhausted)
        budget.append('progress.jsonl', (json.dumps(value, sort_keys=True) + '\n').encode())
        print(json.dumps(value, sort_keys=True), flush=True)

    try:
        progress('ONE_SPATIAL_CPU_CONTINUATION_STARTED')
        for record, row in read_rows(launch['manifest_path'], first_table, launch['manifest_sha256']):
            if row['status'] == 'not_processed_time_limit' and row['decoded_512_RGB'] == 'False':
                if time.monotonic() >= deadline:
                    time_exhausted = True
                if not time_exhausted:
                    attempted += 1
                    try:
                        measurements, failures, error = original.inspect_generated(record, deadline, np, Image)
                        # First timeout row has no completed measurements; preserve only its identities.
                        row = {key: row[key] for key in original.FIELDS[:7]}
                        row.update(hash_verified=False, decoded_512_RGB=False)
                        row.update(measurements)
                        row['status'] = '|'.join(failures) if failures else 'OK'
                        row['error'] = error
                        completed += int(row['decoded_512_RGB'])
                    except original.TimeBudgetExceeded:
                        time_exhausted = True
            add_counts(counts, row)
            table.add(row)
            if counts['records'] % 1000 == 0:
                table.flush()
                progress('ONE_SPATIAL_CPU_CONTINUATION_PROGRESS')
        table.flush()
    except original.OutputBudgetExceeded:
        output_exhausted = True
    except Exception as error:
        source_errors.append(type(error).__name__ + ': ' + str(error)[:200])
        try:
            table.flush()
        except original.OutputBudgetExceeded:
            output_exhausted = True
    groups = [dict(decoded_RGB_pixel_sha256=key, query_count=value[0],
        distinct_file_sha256_count=len(value[1]), identical_pixels_different_file_bytes=len(value[1]) > 1)
        for key, value in sorted(counts['pixels'].items()) if value[0] > 1]
    if not output_exhausted:
        duplicate_table = original.ChunkedCSV(budget, 'pixel_duplicate_groups.csv.gz',
            ['decoded_RGB_pixel_sha256', 'query_count', 'distinct_file_sha256_count', 'identical_pixels_different_file_bytes'])
        try:
            for group in groups:
                duplicate_table.add(group)
            duplicate_table.flush()
        except original.OutputBudgetExceeded:
            output_exhausted = True
    for path, digest in pins.items():
        try:
            checked_sha(path, digest)
        except Exception as error:
            source_errors.append('Source changed: ' + str(error)[:200])
    elapsed = time.monotonic() - start
    time_exhausted |= elapsed >= original.MAX_SECONDS
    coverage = (counts['records'] == table.rows_written == counts['trusted'] == expected_count
        and not counts['failures'] and not source_errors and not time_exhausted and not output_exhausted)
    summary = dict(first_summary)
    summary.update(schema=SCHEMA, utc=original.utc_now(), continuation_count=1, maximum_continuations=1,
        status='COMPLETE_TECHNICAL_COVERAGE' if coverage else 'AUDIT_RECORDED_WITH_FAILURES_OR_INCOMPLETE_COVERAGE',
        complete_technical_coverage=coverage,
        source=dict(manifest_path=launch['manifest_path'], manifest_sha256=launch['manifest_sha256'],
            tool_path=str(Path(__file__).resolve()), tool_sha256=original.sha_file(__file__),
            measurement_tool_path=str(Path(original.__file__).resolve()),
            measurement_tool_sha256=original.sha_file(original.__file__),
            registration=first_summary['source']['registration'], immutable_input_sha256=pins),
        segments=[dict(index=0, output_directory=launch['output_directory'],
                summary_sha256=first_summary_sha256, table_sha256=first_table_sha256,
                tool_sha256=first_summary['source']['tool_sha256'], seconds=first_summary['budget']['seconds'],
                decoded=first_summary['decoded_512_RGB']),
            dict(index=1, output_directory=str(output), tool_sha256=original.sha_file(__file__),
                measurement_tool_sha256=original.sha_file(original.__file__), seconds=elapsed,
                originally_pending=pending, images_attempted=attempted, newly_decoded=completed)],
        original_process_termination=terminated, original_launch_receipt=str(Path(launch_receipt).resolve()),
        original_launch_receipt_sha256=launch_sha256,
        visited_records=counts['records'], generated_files_attempted=first_summary['generated_files_attempted'] + attempted,
        per_query_rows_written=table.rows_written, per_query_rows_unwritten=expected_count - table.rows_written,
        decoded_512_RGB=counts['decoded'], hash_verified_decoded_without_other_failures=counts['trusted'],
        spatial_flags_unknown=expected_count - counts['decoded'], failures=dict(counts['failures']),
        source_errors=source_errors, decoded_flag_counts={name: counts['flags'][name] for name in FLAGS},
        hash_verified_flag_counts={name: counts['trusted_flags'][name] for name in FLAGS},
        decoded_flag_fractions_full_denominator={name: counts['flags'][name] / expected_count for name in FLAGS},
        hash_verified_flag_fractions_full_denominator={name: counts['trusted_flags'][name] / expected_count for name in FLAGS},
        unknown_fraction_full_denominator=(expected_count - counts['decoded']) / expected_count,
        decoded_pixel_duplicate_groups=len(groups),
        duplicate_groups_with_different_file_bytes=sum(g['identical_pixels_different_file_bytes'] for g in groups),
        duplicate_group_queries=sum(g['query_count'] for g in groups),
        budget=dict(maximum_seconds=original.MAX_SECONDS, seconds=elapsed,
            maximum_output_bytes=original.MAX_OUTPUT_BYTES, output_bytes_before_summary=budget.used,
            time_budget_exhausted=time_exhausted, output_budget_exhausted=output_exhausted,
            memory_preflight=memory, thread_environment={name: os.environ[name] for name in original.THREAD_VARIABLES}),
        continuation_policy=dict(only_exact_original_timeout_rows_read=True, completed_images_read_again=False,
            first_artifacts_modified=False, new_images_generated=False, new_thresholds=False,
            automatic_retry=False, further_continuation_allowed=False,
            cached_completed_file_SHA_revalidated_from_pinned_first_table_not_rehashed_from_images=True))
    budget.json('summary.json', summary)
    try:
        progress('ONE_SPATIAL_CPU_CONTINUATION_FINISHED')
    except original.OutputBudgetExceeded:
        pass
    assert sum(path.stat().st_size for path in output.iterdir() if path.is_file()) <= original.MAX_OUTPUT_BYTES
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-receipt', required=True)
    parser.add_argument('--launch-sha256', required=True)
    parser.add_argument('--first-summary-sha256', required=True)
    parser.add_argument('--first-table-sha256', required=True)
    parser.add_argument('--expected-count', type=int, default=16600)
    args = parser.parse_args()
    result = finish(**vars(args))
    print(json.dumps(dict(status=result['status'], full_manifest_denominator=result['full_manifest_denominator'])), flush=True)


if __name__ == '__main__':
    main()
