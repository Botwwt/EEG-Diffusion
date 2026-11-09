"""CPU-only streaming export: ten SHA-verified R4 means, T-only 17 electrodes."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from run_BrainHIVE_trialset_fastscreen_20261027_v1 import read_index

ROOT = Path(__file__).resolve().parent
SCHEMA = 'BrainHIVE_raw_neural_recoverability_20261027_v1'
BINDINGS_SHA256 = '7acfb0bdc3f6627150f8f320e21ca6cdc7f04ff9114ba498496ba4f8c458fa66'
SUBJECTS = tuple(f'sub-{i:02d}' for i in range(1, 11))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.pending')
    assert not path.exists() and not temporary.exists()
    with temporary.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.rename(path)


def extract_block(source, original_rows, channels):
    assert source.ndim == 3 and source.dtype == torch.float32 and source.device.type == 'cpu'
    assert len(channels) == 17 and len(set(channels)) == 17 and min(channels) >= 0 and max(channels) < source.shape[1]
    assert len(original_rows) == len(set(original_rows))
    values = source[torch.tensor(original_rows, dtype=torch.long)][:, channels, :].contiguous()
    assert torch.isfinite(values).all()
    return values.numpy()


def self_test():
    source = torch.arange(30 * 63 * 5, dtype=torch.float32).reshape(30, 63, 5)
    indices = [4, 1, 29]
    channels = list(range(46, 63))
    value = extract_block(source, indices, channels)
    assert value.shape == (3, 17, 5)
    assert np.array_equal(value, source[indices][:, channels].numpy())
    try:
        extract_block(source, [1, 1], channels)
    except AssertionError:
        pass
    else:
        raise AssertionError('Duplicated stimulus row was accepted')
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(status='RAW_NEURAL_PREP_SYNTHETIC_CHECKS_PASSED', checks=3, GPU_used=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bindings', 'index', 'split', 'output', 'path-overrides'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--max-seconds', type=float, default=1200)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    assert all((args.bindings, args.index, args.split, args.output)) and 0 < args.max_seconds <= 1200
    assert not args.output.exists(), 'Preserve previous export and failures'
    args.output.mkdir(parents=True)
    started = time.monotonic()
    torch.set_num_threads(2)
    from diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 import process_rss
    baseline_rss = process_rss()
    peak_extra_CPU_bytes = 0
    assert not torch.cuda.is_initialized()
    assert sha(args.bindings) == BINDINGS_SHA256
    bindings = json.loads(args.bindings.read_bytes())
    assert sha(args.split) == bindings['split_manifest_sha256']
    split = json.loads(args.split.read_bytes())
    items = read_index(args.index)
    assert len(items) == 13230 and [row['row'] for row in items] == list(range(13230))
    original_rows = [int(row['source_row']) for row in items]
    assert set(original_rows) == set(split['image_rows']['T']) and len(set(original_rows)) == 13230
    concepts = {}
    for item in items:
        assert item['fold'] == int(split['oof_fold_by_concept'][item['concept_id']])
        concepts.setdefault(item['concept_id'], []).append(item)
    assert len(concepts) == 1323 and all(len(group) == 10 for group in concepts.values())
    channels = bindings['selected_indices']
    assert channels == list(range(46, 63)) and len(bindings['selected_channels']) == 17
    overrides = json.loads(args.path_overrides.read_bytes()) if args.path_overrides else {}
    assert set(overrides) <= set(SUBJECTS)
    required = 10 * 13230 * 17 * 250 * 4 + (512 << 20)
    assert shutil.disk_usage(args.output).free >= required, 'Reserve complete ten-subject compact output'
    subjects = {}
    initial_index_sha = sha(args.index)
    try:
        for subject in SUBJECTS:
            assert time.monotonic() - started < args.max_seconds
            registered = bindings['subjects'][subject]
            assert registered['train_repeats'] == 4
            path = Path(overrides.get(subject, registered['train_path']))
            assert sha(path) == registered['train_sha256'], 'Trusted whole source SHA before torch decoding'
            source = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
            assert torch.is_tensor(source) and source.shape == (16540, 63, 250) and source.dtype == torch.float32
            destination = args.output / f'{subject}.npy'
            export = np.lib.format.open_memmap(destination, mode='w+', dtype=np.float32, shape=(13230, 17, 250))
            for start in range(0, 13230, 128):
                assert time.monotonic() - started < args.max_seconds
                block = extract_block(source, original_rows[start:start+128], channels)
                export[start:start+len(block)] = block
                peak_extra_CPU_bytes = max(peak_extra_CPU_bytes, process_rss()-baseline_rss)
                assert peak_extra_CPU_bytes <= 10 << 30
            export.flush()
            del export, source
            with destination.open('rb') as stream:
                os.fsync(stream.fileno())
            gc.collect()
            assert sha(path) == registered['train_sha256']
            subjects[subject] = dict(path=str(destination.resolve()), sha256=sha(destination), shape=[13230,17,250],
                source_path=str(path), registered_source_path=registered['train_path'], source_sha256=registered['train_sha256'],
                source_shape=[16540,63,250], selected_channels=channels, repeat_protocol='physical_mean4')
            print(json.dumps(dict(stage='RAW_NEURAL_SUBJECT_COMPACT_COMPLETE', subject=subject,
                subjects_complete=len(subjects), expected_subjects=10, seconds=time.monotonic()-started)), flush=True)
        index_path = args.output / 'index.jsonl'
        index_path.write_text(''.join(json.dumps(row) + '\n' for row in items), encoding='utf-8')
        assert sha(args.index) == initial_index_sha and not torch.cuda.is_initialized()
        atomic(args.output / 'completion.json', dict(status='COMPLETE_RAW_NEURAL_T17_ALL10_INPUTS', schema=SCHEMA,
            subjects=subjects, prediction_index=str(index_path.resolve()), prediction_index_sha256=sha(index_path),
            original_index_sha256=initial_index_sha, input_bindings_sha256=BINDINGS_SHA256,
            split_sha256=sha(args.split), selected_channels=bindings['selected_channels'],
            selected_indices=channels, source_rows=original_rows, rows=13230, concepts=1323,
            only_train_paths_opened=True, test_R80_opened=False, D_V_values_exported=False,
            no_visual_feature_or_latent_loaded=True, CUDA_initialized=False,
            source=str(Path(__file__).resolve()), source_sha256=sha(__file__), seconds=time.monotonic()-started,
            peak_extra_CPU_bytes=peak_extra_CPU_bytes, maximum_extra_CPU_bytes=10 << 30,
            existing_preprocessing_MVNN_training_only_established=False,
            independent_session_metadata_available=False))
    except BaseException as error:
        atomic(args.output / 'failure.json', dict(status='INCOMPLETE_RAW_NEURAL_EXPORT_PRESERVED',
            error=repr(error), completed_subjects=list(subjects), expected_subjects=list(SUBJECTS),
            CUDA_initialized=torch.cuda.is_initialized(), seconds=time.monotonic()-started))
        raise


if __name__ == '__main__':
    main()
