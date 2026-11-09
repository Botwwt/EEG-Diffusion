"""CPU-only export of verified physical R4 trials; no test/D/V arrays are read."""
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
WORK = Path('/root/brainhive_reproduction_20261020_v1')
OUT = Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/trialset_fastscreen_20261027_v1/inputs')
RAW = Path('/root/autodl-tmp/atm_published_training_trials_20261016_v2/preprocessed/sub-01/preprocessed_eeg_training.npy')
RAW_SHA = 'c54cf2e8a4f0c122d79bb7d98ddf04698f992d292f3e26ae59bed97230944f63'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, obj):
    p = Path(path)
    t = p.with_name(p.name + '.pending')
    t.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    t.replace(p)


def main():
    start = time.monotonic()
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    OUT.mkdir(parents=True, exist_ok=True)
    assert not any(OUT.iterdir()), 'Preserve earlier exports; no silent overwrite'
    inputs = WORK / 'data/actual_input_bindings.json'
    contract_path = WORK / 'data/component_input_contract_v1.json'
    split_path = ROOT / 'configs/split_manifest.json'
    manifest_path = ROOT / 'audit/image_manifest.csv'
    bindings = json.loads(inputs.read_text())
    contract = json.loads(contract_path.read_text())
    split = json.loads(split_path.read_text())
    assert sha(split_path) == bindings['split_manifest_sha256']
    assert sha(manifest_path) == split['image_manifest_sha256'] == bindings['image_manifest_sha256']
    subject = bindings['subjects']['sub-01']
    indices = bindings['selected_indices']
    assert len(indices) == len(set(indices)) == 17 and bindings['selected_channels'][-3:] == ['O1', 'Oz', 'O2']
    assert RAW.is_file() and RAW.stat().st_size == 4168083005
    assert sha(RAW) == RAW_SHA, 'Complete trusted source identity before pickle decoding'
    print(json.dumps(dict(status='SOURCE_SHA_VERIFIED', seconds=time.monotonic()-start)), flush=True)
    source = np.load(RAW, allow_pickle=True)
    raw = source['preprocessed_eeg_data']
    assert raw.shape == (16540, 4, 63, 250) and raw.dtype == np.float32
    assert sha(subject['train_path']) == subject['train_sha256']
    old_mean = torch.load(subject['train_path'], weights_only=True, mmap=True, map_location='cpu')
    assert tuple(old_mean.shape) == (16540, 63, 250)
    rows = split['image_rows']['T']
    assert len(rows) == 13230 and len(set(rows)) == 13230
    assert set(rows).isdisjoint(split['image_rows']['D'] + split['image_rows']['V'])
    image_rows = {int(r['array_row']): r for r in csv.DictReader(manifest_path.open()) if r['dataset_split'] == 'train'}
    outpath = OUT / 'trials.npy'
    arr = np.lib.format.open_memmap(str(outpath), mode='w+', dtype=np.float32, shape=(len(rows), 4, 17, 250))
    unequal = 0
    # Entire original training mean replay, not only the exported T subset.
    for lo in range(0, 16540, 64):
        assert time.monotonic() - start < 600
        block = raw[lo:lo+64]
        assert np.isfinite(block).all()
        assert np.array_equal(block.mean(1), old_mean[lo:lo+len(block)].numpy()), 'Physical R4 / existing mean mismatch'
    for lo in range(0, len(rows), 64):
        block = raw[rows[lo:lo+64]][:, :, indices, :]
        arr[lo:lo+len(block)] = block
        unequal += int(np.any(block[:, 1:] != block[:, :1], axis=(1, 2, 3)).sum())
    arr.flush()
    del arr
    assert unequal == len(rows), 'Every exported stimulus must have genuine distinct physical trials'
    target_path = Path(contract['target_cache_path'])
    assert sha(target_path) == contract['target_cache_sha256']
    targets = torch.load(target_path, weights_only=True, mmap=True, map_location='cpu')[rows].numpy().copy()
    assert targets.shape == (13230, 1024) and targets.dtype == np.float32 and np.isfinite(targets).all()
    np.save(OUT/'targets.npy', targets, allow_pickle=False)
    index = []
    for compact_row, source_row in enumerate(rows):
        m = image_rows[source_row]
        index.append(dict(row=compact_row, target_row=compact_row, source_row=source_row,
                          array_row=source_row, concept_id=m['concept_id'], stimulus_id=m['stimulus_id'],
                          fold=int(split['oof_fold_by_concept'][m['concept_id']]), subject_id='sub-01',
                          partition='T', image_zip=m['image_zip'], image_member=m['image_member'],
                          image_sha256=m['image_sha256']))
    (OUT/'index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in index))
    # Locked before fitting or reading held metrics: eight complete concepts.
    chosen = []
    for fold in (0, 1):
        concepts = {r['concept_id'] for r in index if r['fold'] == fold}
        ordered = sorted(concepts, key=lambda c: hashlib.sha256(('trialset-panel-1729:'+c).encode()).hexdigest())
        chosen += ordered[:4]
    panel = [r for r in index if r['concept_id'] in set(chosen)]
    assert len(panel) == 80
    write(OUT/'screen_panel.json', dict(status='FIXED_BEFORE_FIT', subject='sub-01', selection='SHA256 trialset-panel-1729:concept; four concepts per fold; all ten stimuli', rows=panel))
    files = {p.name:dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size) for p in OUT.iterdir() if p.is_file()}
    receipt = dict(status='PHYSICAL_R4_T_INPUTS_READY', source=str(RAW), source_sha256=RAW_SHA,
                   source_shape=list(raw.shape), export_shape=[13230,4,17,250],
                   subject='sub-01', channels=bindings['selected_channels'], selected_indices=indices,
                   complete_original_mean_replay_rows=16540, mean_replay_exact=True,
                   genuine_distinct_trial_rows=unequal, T_rows=len(rows), T_concepts=len({r['concept_id'] for r in index}),
                   input_bindings_sha256=sha(inputs), input_contract_sha256=sha(contract_path),
                   split_sha256=sha(split_path), image_manifest_sha256=sha(manifest_path),
                   source_after_sha256=sha(RAW), files=files, seconds=time.monotonic()-start,
                   CUDA_initialized=torch.cuda.is_initialized(), independent_session_metadata_available=False,
                   upstream_MVNN_training_only_established=False, official_test_read=False,
                   claim_limit='single-subject exploratory T-only screen; not full confirmation')
    assert receipt['source_after_sha256'] == RAW_SHA and not receipt['CUDA_initialized']
    write(OUT/'completion.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
