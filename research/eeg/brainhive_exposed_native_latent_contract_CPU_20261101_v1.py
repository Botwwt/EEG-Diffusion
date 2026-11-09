"""Separate truthful exposed full-TRAIN input stage; legacy official gates untouched."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

STATUS = 'REVIEWED_EXPOSED_FULL16540_NATIVE_VAE_INPUT_STAGE_ONLY'
GPU = 'GPU-0a99d971-a46c-6e81-f157-c33243c6d57a'
SHAPE = (16540, 4, 64, 64)
PROFILE_INDICES = (0, 1, 2, 3, 16536, 16537, 16538, 16539)


def read(path):
    return json.loads(Path(path).read_bytes())


def ref(path):
    path = Path(path)
    assert path.is_file() and not path.is_symlink(), str(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest)


def check(pin):
    assert ref(pin['path']) == pin
    return Path(pin['path'])


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def validate_scope(s):
    assert s['status'] == STATUS and s['host'] == 'old' and s['GPU_UUID'] == GPU
    assert s['official200_previously_exposed'] is True and s['official_results_used_for_previous_direction'] is True
    assert s['independent_H'] is False and s['official_test_image_payloads_allowed'] is False
    assert s['component_fit_or_benchmark_admitted'] is False and s['new_training_seed'] is False
    assert s['shape'] == list(SHAPE) and s['batch_size'] == 4 and s['chunk_size'] == 256
    assert s['dtype'] == 'float32' and s['mode'] == 'scaled_native_VAE_posterior_mode_no_sampling'
    assert s['preprocessing'] == 'Unchanged author Resize512 BILINEAR, CenterCrop512, RGB[-1,1]'
    assert s['profile_indices'] == list(PROFILE_INDICES)
    assert s['segment_chunk_ranges'] == [[0, 32], [32, 65]]
    assert s['maximum_output_bytes'] == 2 << 30
    assert s['legacy_floor_bytes'] == 50 << 30 and s['disk_buffer_bytes'] == 8 << 30
    assert s['upstream_remaining_budget_reserved_bytes'] == 12 << 30
    assert s['minimum_start_CPU_headroom_bytes'] == 24 << 30
    assert s['minimum_runtime_CPU_headroom_bytes'] == 10 << 30
    assert s['maximum_extra_CPU_bytes'] == 10 << 30 and s['maximum_GPU_bytes'] == 20 << 30
    assert s['per_GPU_job_seconds'] == 900 and s['numerical_guard'] == dict(atol=1e-5, rtol=5e-6)
    assert s['legacy_official_entrypoints_or_gates_modified'] is False
    assert s['legacy_generic_transaction_writer_reused_under_this_separate_review'] is True
    assert s['old_T']['available'] is True
    return s


def setup(path):
    s = validate_scope(read(path))
    for pin in s['source_pins']:
        check(pin)
    review = read(check(s['source_data_resource_review']))
    assert review['status'] == 'PASS_SOURCE_DATA_RESOURCE_REVIEW_FOR_LATENT_STAGE_ONLY'
    assert review['source_pins'] == s['source_pins'] and review['data_review'] == s['data_review']
    sys.path.insert(0, s['repository_root'])
    import brainhive_paired_study_contract_20261025_v1 as C
    C.setup_runtime(dict(runtime_manifest=s['runtime_manifest']))
    sys.path.insert(0, str(Path(path).parent))
    return s


def train_rows(s):
    with check(s['image_manifest']).open(newline='') as stream:
        rows = [dict(r, array_row=int(r['array_row']), image_bytes=int(r['image_bytes']))
                for r in csv.DictReader(stream) if r['dataset_split'] == 'train']
    rows.sort(key=lambda r:r['array_row'])
    assert len(rows) == 16540 and [r['array_row'] for r in rows] == list(range(16540))
    assert len({r['stimulus_id'] for r in rows}) == len({r['image_sha256'] for r in rows}) == 16540
    assert len({r['concept_id'] for r in rows}) == 1654
    assert {r['image_zip'] for r in rows} == set(s['archives'])
    return rows


def input_check(s):
    for pin in (s['data_review'], s['decoder_registration'], s['generator'], s['VAE_config'],
                s['asset_protocol'], s['input_bank'], s['old_T']['receipt']):
        check(pin)
    for pin in s['archives'].values():
        check(pin)
    if s['old_T']['available']:
        check(s['old_T']['bank'])
    return train_rows(s)


def allocated(root):
    total, seen = 0, set()
    for path in Path(root).rglob('*'):
        assert not path.is_symlink()
        if path.is_file():
            value = path.stat()
            key = value.st_dev, value.st_ino
            if key not in seen:
                seen.add(key)
                total += value.st_blocks * 512 if hasattr(value, 'st_blocks') else value.st_size
    return total


def resources(s, starting=False, extra_bytes=0):
    from brainhive_functional_training_contract_v2 import resource_headroom
    headroom = resource_headroom()['effective_headroom_bytes']
    assert headroom >= s['minimum_start_CPU_headroom_bytes' if starting else 'minimum_runtime_CPU_headroom_bytes']
    root = Path(s['output_root'])
    used = allocated(root) if root.exists() else 0
    assert used + extra_bytes <= s['maximum_output_bytes']
    required = (s['maximum_output_bytes']-used+s['legacy_floor_bytes']+s['disk_buffer_bytes']+
                s['upstream_remaining_budget_reserved_bytes'])
    free = shutil.disk_usage(root if root.exists() else root.parent).free
    assert free >= required
    return dict(effective_CPU_headroom_bytes=headroom, used_output_bytes=used,
                free_bytes=free, required_remaining_bytes=required, physical_reservation=False)


def rss():
    return int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines()
                    if line.startswith('VmHWM:'))) * 1024


def require_wrapper():
    assert 'run_gpu_job_v2.py' in Path('/proc', str(os.getppid()), 'cmdline').read_bytes().decode()
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'], text=True).strip() == GPU
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
