"""Small synthetic image contracts only; no actual EEG, targets, models or scan."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import audit_BrainHIVE_spatial_image_integrity_CPU_v1 as audit


def check():
    # Importing the entry point alone must not import a tensor/model/image stack.
    assert not any(name in sys.modules for name in ('torch', 'numpy', 'PIL.Image'))
    import numpy as np
    from PIL import Image
    boundary = np.full((512, 512, 3), 99, dtype=np.uint8)
    boundary[::2] = 101
    boundary_statistics = audit.pixel_statistics(boundary, np)
    assert boundary_statistics['near_spatial_constant'] is True
    for channel in 'RGB':
        assert boundary_statistics[f'spatial_std_{channel}_0_1_ddof0'] == 1 / 255
    with tempfile.TemporaryDirectory(prefix='hive_spatial_CPU_') as temporary:
        root = Path(temporary)
        purple = np.zeros((512, 512, 3), dtype=np.uint8)
        purple[:, :, 0] = purple[:, :, 2] = 255
        x = np.linspace(0, 255, 512, dtype=np.uint8)
        gradient = np.broadcast_to(x[None, :, None], (512, 512, 3)).copy()
        checker = np.repeat((((np.indices((512, 512)).sum(0) % 2) * 255)
                             .astype(np.uint8))[:, :, None], 3, axis=2)
        images = [('purple_png', purple, 'PNG'), ('gradient', gradient, 'PNG'),
                  ('checkerboard', checker, 'PNG'), ('purple_bmp', purple, 'BMP'),
                  ('hash_mismatch', gradient, 'PNG'),
                  ('wrong_dimensions', purple[:16, :16], 'PNG')]
        records = []
        for name, pixels, format_name in images:
            path = root / (name + '.' + format_name.lower())
            Image.fromarray(pixels, mode='RGB').save(path, format=format_name)
            records.append(dict(query_id='sub01/' + name + '/R4', subject='sub01',
                stimulus_id=name + '.jpg', generated_path=str(path), generated_sha256=audit.sha_file(path),
                score_target_path=str(root / 'FORBIDDEN_TARGET_DO_NOT_READ'),
                EEG_path=str(root / 'FORBIDDEN_EEG_DO_NOT_READ')))
        corrupt = root / 'corrupt.webp'
        corrupt.write_bytes(b'not an image')
        records.append(dict(query_id='sub01/corrupt/R4', subject='sub01', stimulus_id='corrupt.jpg',
                            generated_path=str(corrupt), generated_sha256=audit.sha_file(corrupt)))
        records.append(dict(query_id='sub01/missing/R4', subject='sub01', stimulus_id='missing.jpg',
                            generated_path=str(root / 'missing.webp'), generated_sha256='a' * 64))
        next(r for r in records if r['stimulus_id'] == 'hash_mismatch.jpg')['generated_sha256'] = '0' * 64
        manifest = root / 'manifest.jsonl'
        manifest.write_text(''.join(json.dumps(row) + '\n' for row in records), encoding='utf-8')
        allowed = {r['generated_path'] for r in records}
        observed = []
        original_reader = audit.read_generated_bytes
        def guarded_reader(path, deadline):
            assert str(path) in allowed
            observed.append(str(path))
            return original_reader(path, deadline)
        registration = Path(__file__).parent / 'configs/BrainHIVE_spatial_image_integrity_CPU_v1_preregistration.json'
        output = root / 'out'
        with patch.object(audit, 'read_generated_bytes', guarded_reader):
            result = audit.audit(manifest, output, expected_count=8,
                registration=registration, registration_sha256=audit.sha_file(registration))
        with gzip.open(output / 'per_query.csv.gz', 'rt', newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == result['full_manifest_denominator'] == 8
        assert result['per_query_rows_written'] == 8
        assert result['decoded_512_RGB'] == 5
        assert result['hash_verified_decoded_without_other_failures'] == 4
        assert result['spatial_flags_unknown'] == 3
        assert not result['complete_technical_coverage']
        for name in ('missing', 'corrupt_or_decode_failure', 'hash_mismatch', 'dimension_failure'):
            assert result['failures'][name] == 1
        assert set(observed) == allowed and len(observed) == 8
        by_stimulus = {r['stimulus_id']: r for r in rows}
        png, bmp = by_stimulus['purple_png.jpg'], by_stimulus['purple_bmp.jpg']
        assert png['actual_file_sha256'] != bmp['actual_file_sha256']
        assert png['decoded_RGB_pixel_sha256'] == bmp['decoded_RGB_pixel_sha256']
        for row in (png, bmp):
            assert float(row['legacy_global_RGB_std']) > .4
            assert row['legacy_global_blank_proxy'] == 'False'
            assert row['exact_spatial_constant'] == row['near_spatial_constant'] == 'True'
            for channel in 'RGB':
                assert float(row[f'spatial_ptp_{channel}_0_255']) == 0
                assert float(row[f'spatial_std_{channel}_0_1_ddof0']) == 0
        assert result['decoded_flag_counts']['exact_spatial_constant'] == 2
        assert result['decoded_flag_counts']['near_spatial_constant'] == 2
        assert result['decoded_flag_fractions_full_denominator']['exact_spatial_constant'] == .25
        assert result['unknown_fraction_full_denominator'] == 3 / 8
        checkered = by_stimulus['checkerboard.jpg']
        assert float(checkered['horizontal_adjacent_abs_mean_0_1']) == 1
        assert float(checkered['vertical_adjacent_abs_mean_0_1']) == 1
        assert float(checkered['interior_four_neighbour_Laplacian_abs_mean_0_1']) == 4
        assert float(checkered['all_RGB_le_1_dark_pixel_fraction']) == .5
        assert float(checkered['all_RGB_ge_254_bright_pixel_fraction']) == .5
        assert float(by_stimulus['gradient.jpg']['horizontal_adjacent_abs_mean_0_1']) < .01
        mismatch = by_stimulus['hash_mismatch.jpg']
        assert mismatch['hash_verified'] == 'False' and mismatch['decoded_512_RGB'] == 'True'
        assert mismatch['status'] == 'hash_mismatch'
        assert mismatch['decoded_RGB_pixel_sha256'] == by_stimulus['gradient.jpg']['decoded_RGB_pixel_sha256']
        for name in ('missing', 'corrupt', 'wrong_dimensions'):
            assert by_stimulus[name + '.jpg']['exact_spatial_constant'] == ''
        assert result['duplicate_groups_with_different_file_bytes'] == 1
        with gzip.open(output / 'pixel_duplicate_groups.csv.gz', 'rt', newline='') as stream:
            groups = list(csv.DictReader(stream))
        assert len(groups) == 2
        assert sum(g['identical_pixels_different_file_bytes'] == 'True' for g in groups) == 1
        assert sum(p.stat().st_size for p in output.iterdir()) <= audit.MAX_OUTPUT_BYTES
        try:
            audit.audit(manifest, output, expected_count=8)
        except FileExistsError:
            pass
        else:
            raise AssertionError('Existing output directory was accepted')
        # Wrong full denominator prevents image reads; all records remain unknown.
        with patch.object(audit, 'read_generated_bytes', side_effect=AssertionError('Must not read images')):
            mismatch_count = audit.audit(manifest, root / 'wrong_count', expected_count=9)
        assert mismatch_count['full_manifest_denominator'] == 8
        assert mismatch_count['expected_queries'] == 9
        assert mismatch_count['decoded_512_RGB'] == 0
        assert mismatch_count['failures']['not_processed_count_mismatch'] == 8
        assert mismatch_count['per_query_rows_written'] == 8
        # Exact fixed thresholds, SHA binding and small CPU resource limits.
        assert audit.NEAR_THRESHOLD == 1 / 255 and audit.LEGACY_THRESHOLD == .01
        try:
            audit.validate_registration(registration, '0' * 64)
        except ValueError:
            pass
        else:
            raise AssertionError('Wrong preregistration SHA was accepted')
        for changes in ({'max_seconds': 901}, {'max_output_bytes': (16 << 20) + 1}):
            try:
                audit.audit(manifest, root / 'forbidden_budget', expected_count=8, **changes)
            except ValueError:
                pass
            else:
                raise AssertionError('Expanded resource ceiling was accepted')
        with patch.object(audit, 'MIN_AVAILABLE_BYTES', 1 << 100):
            try:
                audit.memory_preflight()
            except RuntimeError:
                pass
            else:
                raise AssertionError('Unavailable CPU memory was accepted')
        assert 'torch' not in sys.modules
        return dict(schema='BrainHIVE_spatial_image_integrity_CPU_contract_v1',
            status='PASS_SMALL_SYNTHETIC_CPU_FIXTURES_ONLY', fixtures=8,
            checked=['constant_purple_global_vs_spatial_counterexample', 'gradient_and_checkerboard_differences',
                'same_RGB_different_file_encoding_SHA', 'missing_corrupt_hash_and_dimension_failures',
                'all_records_and_unknowns_in_full_denominator', 'only_manifest_generated_files_read',
                'new_output_no_overwrite', 'count_mismatch_no_image_read', 'registration_SHA_binding',
                'fixed_thresholds_and_budget_ceiling', 'one_LSB_population_std_inclusive_boundary',
                'small_memory_preflight', 'lazy_import_no_torch'],
            fixture_summary=dict(decoded=5, trusted_decoded=4, unknown=3, denominator=8,
                exact_spatial_constant=2, near_spatial_constant=2,
                purple_legacy_global_RGB_std=float(png['legacy_global_RGB_std']),
                purple_legacy_blank_proxy=False, different_file_bytes_same_pixels_groups=1),
            source_sha256={Path(__file__).name: audit.sha_file(__file__),
                           Path(audit.__file__).name: audit.sha_file(audit.__file__)},
            actual_formal_image_scan_performed=False, EEG_models_targets_or_scores_read=False,
            GPU_used=False, training_or_new_images=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    args = parser.parse_args()
    receipt = check()
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True)
            stream.write('\n')
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
