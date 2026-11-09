"""Two synthetic rows: one cached completion and one exact original timeout."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import finish_BrainHIVE_spatial_integrity_CPU_v1 as continuation
import audit_BrainHIVE_spatial_image_integrity_CPU_v1 as original


def check():
    import numpy as np
    from PIL import Image
    with tempfile.TemporaryDirectory(prefix='hive_spatial_finish_CPU_') as temporary:
        root = Path(temporary)
        records = []
        for index, colour in enumerate(((255, 0, 255), (0, 255, 0))):
            path = root / f'{index}.png'
            Image.new('RGB', (512, 512), colour).save(path)
            records.append(dict(query_id=f'sub01/q{index}/R4', subject='sub01', stimulus_id=f'q{index}.jpg',
                generated_path=str(path), generated_sha256=original.sha_file(path)))
        manifest = root / 'manifest.jsonl'
        manifest.write_text(''.join(json.dumps(r) + '\n' for r in records))
        registration = Path(__file__).parent / 'configs/BrainHIVE_spatial_image_integrity_CPU_v1_preregistration.json'
        registration_sha = original.sha_file(registration)
        inspect = original.inspect_generated
        def first_segment(record, *args):
            if record['query_id'] == records[1]['query_id']:
                raise original.TimeBudgetExceeded('Synthetic original timeout before reading second image')
            return inspect(record, *args)
        first = root / 'first'
        with patch.object(original, 'inspect_generated', first_segment):
            original.audit(manifest, first, expected_count=2, registration=registration, registration_sha256=registration_sha)
        launch = dict(status='ACTUAL_COMPLETE_RISK_SPATIAL_CPU_AUDIT_STARTED', pid=123456789,
            CPU_only=True, new_training=False, new_images=False, new_semantic_scores=False,
            maximum_seconds=900, maximum_output_bytes=16 << 20, expected_queries=2,
            output_directory=str(first), manifest_path=str(manifest), manifest_sha256=original.sha_file(manifest),
            registration_path=str(registration.resolve()), registration_sha256=registration_sha,
            source_sha256={Path(original.__file__).name: original.sha_file(original.__file__)},
            argv=['python', str(Path(original.__file__).resolve()), '--manifest', str(manifest), '--output', str(first),
                '--expected-count', '2', '--registration', str(registration.resolve()), '--registration-sha256', registration_sha])
        launch_path = root / 'launch.json'
        launch_path.write_text(json.dumps(launch))
        arguments = dict(launch_receipt=launch_path, launch_sha256=original.sha_file(launch_path),
            first_summary_sha256=original.sha_file(first / 'summary.json'),
            first_table_sha256=original.sha_file(first / 'per_query.csv.gz'), expected_count=2)
        pins_before = {path: original.sha_file(path) for path in (manifest, launch_path, first / 'summary.json', first / 'per_query.csv.gz')}
        with patch.object(continuation, 'proc_snapshot', return_value=dict(pid=launch['pid'], exists=True, state='R', live=True)):
            try:
                continuation.finish(**arguments)
            except AssertionError:
                pass
            else:
                raise AssertionError('Live original PID was accepted')
        with patch.object(continuation, 'proc_snapshot', return_value=dict(pid=launch['pid'], exists=False, state=None, live=False)):
            bad_arguments = dict(arguments, first_table_sha256='0' * 64)
            try:
                continuation.finish(**bad_arguments)
            except AssertionError:
                pass
            else:
                raise AssertionError('Wrong first-table SHA was accepted')
            observed = []
            def only_pending(record, *args):
                observed.append(record['query_id'])
                assert record['query_id'] == records[1]['query_id'], 'Completed image was re-read'
                return inspect(record, *args)
            with patch.object(original, 'inspect_generated', only_pending):
                result = continuation.finish(**arguments)
            assert observed == [records[1]['query_id']]
            assert result['complete_technical_coverage'] is True
            assert result['full_manifest_denominator'] == result['per_query_rows_written'] == 2
            assert result['decoded_512_RGB'] == result['hash_verified_decoded_without_other_failures'] == 2
            assert result['spatial_flags_unknown'] == 0 and result['failures'] == {}
            assert result['decoded_flag_counts']['exact_spatial_constant'] == 2
            assert result['continuation_count'] == result['maximum_continuations'] == 1
            assert result['segments'][1]['images_attempted'] == result['segments'][1]['newly_decoded'] == 1
            merged_path = first.with_name(first.name + '_finish_once_v1')
            with gzip.open(first / 'per_query.csv.gz', 'rt', newline='') as stream:
                old_rows = list(csv.DictReader(stream))
            with gzip.open(merged_path / 'per_query.csv.gz', 'rt', newline='') as stream:
                merged_rows = list(csv.DictReader(stream))
            assert merged_rows[0] == old_rows[0]
            assert merged_rows[1]['status'] == 'OK'
            assert [r['query_id'] for r in merged_rows] == [r['query_id'] for r in records]
            for path, digest in pins_before.items():
                assert original.sha_file(path) == digest
            try:
                continuation.finish(**arguments)
            except AssertionError:
                pass
            else:
                raise AssertionError('Second continuation/retry was accepted')
        return dict(schema='BrainHIVE_spatial_finish_CPU_contract_v1', status='PASS_TWO_SYNTHETIC_ROWS_ONLY',
            checks=['live_original_PID_rejected', 'first_table_SHA_mismatch_rejected',
                'only_original_timeout_image_read', 'cached_completed_row_preserved_exactly',
                'full_two_row_denominator_merged', 'first_artifact_SHAs_unchanged', 'second_continuation_rejected'],
            source_sha256={Path(__file__).name: original.sha_file(__file__),
                Path(continuation.__file__).name: original.sha_file(continuation.__file__),
                Path(original.__file__).name: original.sha_file(original.__file__)},
            actual_formal_images_scanned=False, GPU_used=False, targets_or_models_read=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    args = parser.parse_args()
    result = check()
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open('x') as stream:
            json.dump(result, stream, sort_keys=True, indent=2)
            stream.write('\n')
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
