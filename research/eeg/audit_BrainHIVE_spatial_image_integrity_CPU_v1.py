"""Read-only saved-image engineering diagnostics; no targets, models or scores.

The per-query table uses concatenated gzip members to keep completed chunks
readable if a fixed time/output budget is exhausted. Failures stay in the full
manifest denominator. Spatial flags are technical proxies, not image quality,
purple-noise, naturalness or noninferiority judgements.
"""
import argparse
import collections
import csv
import datetime
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import time


SCHEMA = 'BrainHIVE_spatial_image_integrity_CPU_v1'
MAX_SECONDS = 900
MAX_OUTPUT_BYTES = 16 << 20
MIN_AVAILABLE_BYTES = 128 << 20
SUMMARY_RESERVE_BYTES = 256 << 10
MAX_IMAGE_BYTES = 8 << 20
MAX_MANIFEST_BYTES = 64 << 20
MAX_MANIFEST_LINE_BYTES = 64 << 10
NEAR_THRESHOLD = 1 / 255
LEGACY_THRESHOLD = .01
THREAD_VARIABLES = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')
FIELDS = [
    'source_line_number', 'source_manifest_sha256', 'query_id', 'subject', 'stimulus_id',
    'generated_path', 'manifest_generated_sha256', 'actual_file_sha256',
    'decoded_RGB_pixel_sha256', 'status', 'error', 'hash_verified', 'decoded_512_RGB',
    'width', 'height', 'original_mode',
    *[f'spatial_ptp_{c}_0_255' for c in 'RGB'],
    *[f'spatial_std_{c}_0_1_ddof0' for c in 'RGB'],
    *[f'spatial_mean_{c}_0_1' for c in 'RGB'],
    'exact_spatial_constant', 'near_spatial_constant',
    'legacy_global_RGB_std', 'legacy_global_blank_proxy',
    'all_RGB_le_1_dark_pixel_fraction', 'all_RGB_ge_254_bright_pixel_fraction',
    'horizontal_adjacent_abs_mean_0_1', 'vertical_adjacent_abs_mean_0_1',
    'interior_four_neighbour_Laplacian_abs_mean_0_1',
]
REGISTRATION_FLAGS = {
    'exact_spatial_constant': 'max over channels of spatial (max-min) equals 0 uint8 levels.',
    'near_spatial_constant': 'max over channels of spatial population standard deviation in [0,1] is <= 1/255.',
    'legacy_global_blank_proxy': 'Global population standard deviation over H,W,RGB in [0,1] is < .01; reported for comparison only.',
    'missing_corrupt_hash_or_dimension_failure': 'Report each failed query and preserve it in the expected denominator.',
}
REGISTRATION_DIAGNOSTICS = [
    'Per-channel spatial mean, range and population standard deviation',
    'Fraction of pixels with all RGB channels <= 1; fraction with all RGB channels >= 254',
    'Mean absolute horizontal and vertical adjacent pixel differences, normalized to [0,1]',
    'Mean absolute four-neighbour Laplacian on valid interior pixels, normalized by 255',
    'Legacy global RGB population standard deviation',
]


class TimeBudgetExceeded(RuntimeError):
    pass


class OutputBudgetExceeded(RuntimeError):
    pass


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def memory_preflight():
    """Available physical memory, bounded by cgroup-v2 headroom when present."""
    bounds = {}
    if os.name == 'nt':
        import ctypes
        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in ('total_phys', 'avail_phys',
                'total_page', 'avail_page', 'total_virtual', 'avail_virtual', 'extended')]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise RuntimeError('Cannot establish available physical memory')
        bounds['available_physical_bytes'] = int(status.avail_phys)
    else:
        values = {}
        for line in Path('/proc/meminfo').read_text().splitlines():
            key, value = line.split(':', 1)
            values[key] = int(value.strip().split()[0]) * 1024
        bounds['available_physical_bytes'] = values['MemAvailable']
        candidates = {Path('/sys/fs/cgroup')}
        cgroup = Path('/proc/self/cgroup')
        if cgroup.exists():
            for line in cgroup.read_text().splitlines():
                if line.startswith('0::'):
                    node = Path('/sys/fs/cgroup') / line[3:].lstrip('/')
                    while node == Path('/sys/fs/cgroup') or Path('/sys/fs/cgroup') in node.parents:
                        candidates.add(node)
                        if node == Path('/sys/fs/cgroup'):
                            break
                        node = node.parent
        for node in candidates:
            limit, current = node / 'memory.max', node / 'memory.current'
            if limit.is_file() and current.is_file():
                raw = limit.read_text().strip()
                if raw != 'max':
                    bounds[str(node) + ':headroom'] = max(0, int(raw) - int(current.read_text()))
    bounds['effective_available_bytes'] = min(bounds.values())
    bounds['minimum_available_bytes'] = MIN_AVAILABLE_BYTES
    if bounds['effective_available_bytes'] < MIN_AVAILABLE_BYTES:
        raise RuntimeError('Insufficient small CPU audit headroom (128 MiB required)')
    return bounds


def validate_registration(path=None, expected_sha256=None):
    if path is None and expected_sha256 is None:
        return None
    if path is None or expected_sha256 is None:
        raise ValueError('Registration path and SHA256 must be provided together')
    actual = sha_file(path)
    if actual != expected_sha256:
        raise ValueError('Registration SHA256 differs from pre-registered value')
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    assert value['schema'] == SCHEMA + '_preregistration'
    assert value['status'] == 'FIXED_SUPPLEMENTARY_ENGINEERING_DEFINITIONS_BEFORE_FIRST_ACTUAL_SCAN'
    assert value['pixel_contract']['resolution'] == [512, 512]
    assert value['pixel_contract']['mode'] == 'RGB'
    assert value['pixel_contract']['resize_or_colour_conversion_for_measurement'] is False
    assert value['fixed_flags'] == REGISTRATION_FLAGS
    assert value['fixed_continuous_diagnostics'] == REGISTRATION_DIAGNOSTICS
    assert value['data_policy']['failed_queries_remain_in_denominator'] is True
    for key in ('target_images_read', 'EEG_or_model_tensors_read', 'semantic_scores_read', 'new_images_generated'):
        assert value['data_policy'][key] is False
    budget = value['execution_budget']
    assert budget['CPU_only'] is True and budget['model_threads'] == 1
    assert budget['maximum_seconds'] == MAX_SECONDS
    assert budget['maximum_output_bytes'] == MAX_OUTPUT_BYTES
    assert budget['new_output_directory_required'] is True
    assert budget['automatic_retry_or_resampling'] is False
    assert value['engineering_interpretation']['primary_DINO_SEED_ObjectF1_unchanged'] is True
    return dict(path=str(Path(path).resolve()), sha256=actual, schema=value['schema'])


def describe_manifest(path, deadline):
    path = Path(path)
    if path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError('Manifest exceeds the fixed 64 MiB metadata bound')
    digest = hashlib.sha256()
    count = 0
    with path.open('rb') as stream:
        for line in stream:
            if time.monotonic() >= deadline:
                raise TimeBudgetExceeded('Deadline exhausted during manifest preflight')
            if len(line) > MAX_MANIFEST_LINE_BYTES:
                raise ValueError('Manifest record exceeds the fixed 64 KiB bound')
            digest.update(line)
            count += 1
    return count, digest.hexdigest()


class OutputBudget:
    def __init__(self, root, maximum):
        self.root, self.maximum, self.used = Path(root), maximum, 0

    def append(self, name, data, summary=False):
        limit = self.maximum if summary else self.maximum - SUMMARY_RESERVE_BYTES
        if self.used + len(data) > limit:
            raise OutputBudgetExceeded('Fixed output budget reached; no budget expansion')
        path = self.root / name
        with path.open('ab' if path.exists() else 'xb') as stream:
            stream.write(data)
        self.used += len(data)

    def json(self, name, value):
        if (self.root / name).exists():
            raise FileExistsError(str(self.root / name))
        data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
        self.append(name, data, summary=True)


class ChunkedCSV:
    def __init__(self, budget, name, fields):
        self.budget, self.name, self.fields = budget, name, fields
        self.rows_written = 0
        self.buffer = io.StringIO(newline='')
        self.writer = csv.DictWriter(self.buffer, fieldnames=fields, extrasaction='raise', lineterminator='\n')
        self.writer.writeheader()
        self.pending = 0

    def add(self, row):
        self.writer.writerow({key: format(value, '.10g') if isinstance(value, float) else value
                              for key, value in row.items()})
        self.pending += 1
        if self.pending == 128:
            self.flush()

    def flush(self):
        if self.buffer.tell():
            self.budget.append(self.name, gzip.compress(self.buffer.getvalue().encode(), compresslevel=6, mtime=0))
            self.rows_written += self.pending
            self.pending = 0
            self.buffer.seek(0)
            self.buffer.truncate(0)


def read_generated_bytes(path, deadline):
    data = bytearray()
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            if time.monotonic() >= deadline:
                raise TimeBudgetExceeded('Fixed time budget reached during generated-file read')
            chunk = stream.read(256 << 10)
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > MAX_IMAGE_BYTES:
                raise ValueError('Generated file exceeds fixed 8 MiB per-image input bound')
            digest.update(chunk)
    return data, digest.hexdigest()


def pixel_statistics(pixels, np):
    """Original uint8 RGB only; no resizing, target access or learned threshold."""
    assert pixels.dtype == np.uint8 and pixels.shape == (512, 512, 3)
    minimum, maximum = pixels.min(axis=(0, 1)), pixels.max(axis=(0, 1))
    ranges = maximum.astype(np.int16) - minimum.astype(np.int16)
    normalized = pixels.astype(np.float64) / 255.
    means = pixels.mean(axis=(0, 1), dtype=np.float64) / 255.
    stds = pixels.std(axis=(0, 1), dtype=np.float64, ddof=0) / 255.
    # Match image_evaluation.quality_statistics FP32 global std exactly.
    legacy_std = float((pixels.astype(np.float32) / 255.).std(ddof=0))
    laplacian = (normalized[:-2, 1:-1] + normalized[2:, 1:-1]
                 + normalized[1:-1, :-2] + normalized[1:-1, 2:]
                 - 4 * normalized[1:-1, 1:-1])
    row = dict(decoded_RGB_pixel_sha256=hashlib.sha256(pixels.tobytes(order='C')).hexdigest(),
        exact_spatial_constant=bool(ranges.max() == 0),
        near_spatial_constant=bool(stds.max() <= NEAR_THRESHOLD),
        legacy_global_RGB_std=legacy_std,
        legacy_global_blank_proxy=bool(legacy_std < LEGACY_THRESHOLD),
        all_RGB_le_1_dark_pixel_fraction=float(np.all(pixels <= 1, axis=-1).mean()),
        all_RGB_ge_254_bright_pixel_fraction=float(np.all(pixels >= 254, axis=-1).mean()),
        horizontal_adjacent_abs_mean_0_1=float(np.abs(normalized[:, 1:] - normalized[:, :-1]).mean()),
        vertical_adjacent_abs_mean_0_1=float(np.abs(normalized[1:] - normalized[:-1]).mean()),
        interior_four_neighbour_Laplacian_abs_mean_0_1=float(np.abs(laplacian).mean()))
    for index, channel in enumerate('RGB'):
        row[f'spatial_ptp_{channel}_0_255'] = int(ranges[index])
        row[f'spatial_std_{channel}_0_1_ddof0'] = float(stds[index])
        row[f'spatial_mean_{channel}_0_1'] = float(means[index])
    return row


def inspect_generated(record, deadline, np, Image):
    row, failures = {}, []
    try:
        content, actual_sha = read_generated_bytes(record['generated_path'], deadline)
        row['actual_file_sha256'] = actual_sha
        row['hash_verified'] = actual_sha == record['generated_sha256']
        if not row['hash_verified']:
            failures.append('hash_mismatch')
    except TimeBudgetExceeded:
        raise
    except FileNotFoundError:
        return row, ['missing'], 'Generated file does not exist'
    except Exception as error:
        return row, ['generated_file_read_failure'], str(error)[:200]
    try:
        with Image.open(io.BytesIO(content)) as image:
            row.update(width=image.width, height=image.height, original_mode=image.mode)
            if image.size != (512, 512):
                return row, failures + ['dimension_failure'], 'Expected original 512x512; no resizing performed'
            if image.mode != 'RGB':
                return row, failures + ['RGB_mode_failure'], 'Expected decoded RGB; no colour conversion performed'
            image.load()
            pixels = np.asarray(image)
            row.update(pixel_statistics(pixels, np))
            row['decoded_512_RGB'] = True
        return row, failures, ''
    except Exception as error:
        return row, failures + ['corrupt_or_decode_failure'], str(error)[:200]


def audit(manifest, output, expected_count=16600, registration=None,
          registration_sha256=None, max_seconds=MAX_SECONDS, max_output_bytes=MAX_OUTPUT_BYTES):
    start = time.monotonic()
    if not 0 < max_seconds <= MAX_SECONDS:
        raise ValueError('Time limit must be positive and at most 900 seconds')
    if not (1 << 20) <= max_output_bytes <= MAX_OUTPUT_BYTES:
        raise ValueError('Output limit must be between 1 and 16 MiB')
    if not isinstance(expected_count, int) or expected_count <= 0:
        raise ValueError('Expected query count must be a positive integer')
    output = Path(output)
    if output.exists():
        raise FileExistsError('A new output directory is required; existing outputs are never overwritten')
    for name in THREAD_VARIABLES:
        os.environ[name] = '1'
    memory = memory_preflight()
    registered = validate_registration(registration, registration_sha256)
    deadline = start + max_seconds
    count, manifest_sha = describe_manifest(manifest, deadline)
    # Heavy dependencies are lazy and CPU-only. Do not import any model/evaluator.
    import numpy as np
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = 512 * 512
    output.mkdir(parents=True, exist_ok=False)
    budget = OutputBudget(output, max_output_bytes)
    table = ChunkedCSV(budget, 'per_query.csv.gz', FIELDS)
    failures, flags, trusted_flags = collections.Counter(), collections.Counter(), collections.Counter()
    seen_queries, seen_paths, pixel_groups = set(), set(), {}
    processed = decoded = trusted = visited = 0
    time_exhausted = output_exhausted = False
    source_errors = []

    def progress(stage):
        value = dict(stage=stage, utc=utc_now(), visited_records=visited, manifest_denominator=count,
                     decoded_512_RGB=decoded, hash_verified_decoded=trusted,
                     per_query_rows_written=table.rows_written, seconds=time.monotonic() - start,
                     time_budget_exhausted=time_exhausted, output_budget_exhausted=output_exhausted)
        budget.append('progress.jsonl', (json.dumps(value, sort_keys=True) + '\n').encode())
        print(json.dumps(value, sort_keys=True), flush=True)

    try:
        progress('SPATIAL_INTEGRITY_CPU_STARTED')
        with Path(manifest).open('rb') as stream:
            for line_number, line in enumerate(stream, 1):
                visited += 1
                row = dict(source_line_number=line_number, source_manifest_sha256=manifest_sha,
                           hash_verified=False, decoded_512_RGB=False)
                issues, error = [], ''
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError('Manifest record is not an object')
                    for key in ('query_id', 'subject', 'stimulus_id', 'generated_path'):
                        if not isinstance(record.get(key), str) or not record[key]:
                            raise ValueError('Missing nonempty ' + key)
                        row[key] = record[key]
                    expected_sha = record.get('generated_sha256')
                    row['manifest_generated_sha256'] = expected_sha
                    if not isinstance(expected_sha, str) or len(expected_sha) != 64 or any(c not in '0123456789abcdef' for c in expected_sha):
                        raise ValueError('Invalid generated_sha256')
                    if not Path(record['generated_path']).is_absolute():
                        raise ValueError('Generated path must be explicit and absolute')
                    if record['query_id'] in seen_queries:
                        issues.append('duplicate_query_id')
                    if record['generated_path'] in seen_paths:
                        issues.append('duplicate_generated_path')
                    seen_queries.add(record['query_id'])
                    seen_paths.add(record['generated_path'])
                except Exception as exception:
                    record = None
                    issues.append('invalid_manifest_record')
                    error = str(exception)[:200]
                if time.monotonic() >= deadline:
                    time_exhausted = True
                if count != expected_count:
                    issues.append('not_processed_count_mismatch')
                elif time_exhausted:
                    issues.append('not_processed_time_limit')
                elif record is not None:
                    try:
                        measurements, new_issues, error = inspect_generated(record, deadline, np, Image)
                        row.update(measurements)
                        issues.extend(new_issues)
                        processed += 1
                    except TimeBudgetExceeded as exception:
                        time_exhausted = True
                        issues.append('not_processed_time_limit')
                        error = str(exception)
                failures.update(issues)
                if row.get('decoded_512_RGB'):
                    decoded += 1
                    is_trusted = bool(row.get('hash_verified')) and not issues
                    trusted += int(is_trusted)
                    for name in ('exact_spatial_constant', 'near_spatial_constant', 'legacy_global_blank_proxy'):
                        flags[name] += int(row[name])
                        trusted_flags[name] += int(row[name] and is_trusted)
                    pixel_sha = row['decoded_RGB_pixel_sha256']
                    group = pixel_groups.setdefault(pixel_sha, [0, set()])
                    group[0] += 1
                    group[1].add(row['actual_file_sha256'])
                row['status'] = '|'.join(issues) if issues else 'OK'
                row['error'] = error
                table.add(row)
                if visited % 1000 == 0:
                    table.flush()
                    progress('SPATIAL_INTEGRITY_CPU_PROGRESS')
        table.flush()
    except OutputBudgetExceeded:
        output_exhausted = True
        # Complete gzip members already on disk remain readable; no silent rows.
    except Exception as exception:
        source_errors.append(type(exception).__name__ + ': ' + str(exception)[:200])
        try:
            table.flush()
        except OutputBudgetExceeded:
            output_exhausted = True

    duplicate_groups = [dict(decoded_RGB_pixel_sha256=key, query_count=value[0],
        distinct_file_sha256_count=len(value[1]), identical_pixels_different_file_bytes=len(value[1]) > 1)
        for key, value in sorted(pixel_groups.items()) if value[0] > 1]
    if not output_exhausted:
        groups = ChunkedCSV(budget, 'pixel_duplicate_groups.csv.gz',
            ['decoded_RGB_pixel_sha256', 'query_count', 'distinct_file_sha256_count',
             'identical_pixels_different_file_bytes'])
        try:
            for group in duplicate_groups:
                groups.add(group)
            groups.flush()
        except OutputBudgetExceeded:
            output_exhausted = True
    try:
        if sha_file(manifest) != manifest_sha:
            source_errors.append('Manifest changed during read-only audit')
        if registered and sha_file(registered['path']) != registered['sha256']:
            source_errors.append('Registration changed during audit')
    except Exception as exception:
        source_errors.append('Source revalidation failed: ' + str(exception)[:200])
    elapsed = time.monotonic() - start
    if elapsed >= max_seconds:
        time_exhausted = True
    complete_rows = table.rows_written == count
    coverage = (count == expected_count and complete_rows and trusted == count and not failures
                and not source_errors and not time_exhausted and not output_exhausted)
    unknown = count - decoded
    summary = dict(schema=SCHEMA, utc=utc_now(), status=(
        'COMPLETE_TECHNICAL_COVERAGE' if coverage else 'AUDIT_RECORDED_WITH_FAILURES_OR_INCOMPLETE_COVERAGE'),
        complete_technical_coverage=coverage,
        source=dict(manifest_path=str(Path(manifest).resolve()), manifest_sha256=manifest_sha,
                    tool_path=str(Path(__file__).resolve()), tool_sha256=sha_file(__file__),
                    registration=registered),
        expected_queries=expected_count, full_manifest_denominator=count,
        manifest_count_matches_expected=count == expected_count,
        visited_records=visited, generated_files_attempted=processed,
        per_query_rows_written=table.rows_written, per_query_rows_unwritten=count - table.rows_written,
        decoded_512_RGB=decoded, hash_verified_decoded_without_other_failures=trusted,
        spatial_flags_unknown=unknown, failures=dict(failures), source_errors=source_errors,
        decoded_flag_counts={name: flags[name] for name in REGISTRATION_FLAGS if name in (
            'exact_spatial_constant', 'near_spatial_constant', 'legacy_global_blank_proxy')},
        hash_verified_flag_counts={name: trusted_flags[name] for name in (
            'exact_spatial_constant', 'near_spatial_constant', 'legacy_global_blank_proxy')},
        decoded_flag_fractions_full_denominator={name: flags[name] / count if count else None for name in (
            'exact_spatial_constant', 'near_spatial_constant', 'legacy_global_blank_proxy')},
        hash_verified_flag_fractions_full_denominator={name: trusted_flags[name] / count if count else None for name in (
            'exact_spatial_constant', 'near_spatial_constant', 'legacy_global_blank_proxy')},
        unknown_fraction_full_denominator=unknown / count if count else None,
        decoded_pixel_duplicate_groups=len(duplicate_groups),
        duplicate_groups_with_different_file_bytes=sum(g['identical_pixels_different_file_bytes'] for g in duplicate_groups),
        duplicate_group_queries=sum(g['query_count'] for g in duplicate_groups),
        fixed_definitions=dict(resolution=[512, 512], required_mode='RGB', resize=False,
            colour_conversion=False, spatial_std_ddof=0, near_spatial_constant_max_std=NEAR_THRESHOLD,
            legacy_global_std_dtype='float32', legacy_global_blank_threshold_strict_less_than=LEGACY_THRESHOLD,
            dark_pixel='all three uint8 RGB channels <= 1', bright_pixel='all three uint8 RGB channels >= 254',
            Laplacian='absolute four-neighbour Laplacian averaged over valid interior pixels and RGB'),
        budget=dict(maximum_seconds=max_seconds, seconds=elapsed, maximum_output_bytes=max_output_bytes,
                    time_budget_exhausted=time_exhausted, output_budget_exhausted=output_exhausted,
                    output_bytes_before_summary=budget.used, memory_preflight=memory,
                    thread_environment={name: os.environ[name] for name in THREAD_VARIABLES}),
        read_policy=dict(generated_files_only=True, target_images_read=False, semantic_scores_read=False,
                         EEG_or_models_read=False, new_images_generated=False, retries_or_resampling=False),
        interpretation=dict(engineering_flags_only=True, human_ratings=False,
            purple_noise_rate_identified=False, naturalness_or_noise_classification=False,
            perceptual_noninferiority_established=False, primary_semantic_metrics_changed=False,
            failed_queries_remain_in_full_denominator=True,
            decoded_flags_include_hash_mismatch_for_diagnosis=True,
            hash_verified_counts_separately_reported=True,
            pixel_duplicates_do_not_establish_file_reuse_or_bad_quality=True))
    budget.json('summary.json', summary)
    try:
        progress('SPATIAL_INTEGRITY_CPU_FINISHED')
    except OutputBudgetExceeded:
        pass  # Reserved summary exists even when no progress bytes remain.
    actual_output_bytes = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    assert actual_output_bytes <= max_output_bytes
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--expected-count', type=int, default=16600)
    parser.add_argument('--registration')
    parser.add_argument('--registration-sha256')
    parser.add_argument('--max-seconds', type=float, default=MAX_SECONDS)
    parser.add_argument('--max-output-bytes', type=int, default=MAX_OUTPUT_BYTES)
    args = parser.parse_args()
    result = audit(**vars(args))
    print(json.dumps(dict(status=result['status'], summary=str(Path(args.output) / 'summary.json'))), flush=True)


if __name__ == '__main__':
    main()
