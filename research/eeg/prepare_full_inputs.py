"""Verified, resumable server-only recovery of missing full EEG inputs.

Uses immutable author assets already identified by the official HF tree. Source
pickles are never loaded before their complete published LFS SHA256 passes.
"""
from pathlib import Path
import concurrent.futures as futures
import fcntl
import hashlib
import json
import os
import shutil
import threading
import time
import traceback
import urllib.request

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
OUT = Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/datasets')
STAGE = OUT / 'staging'
REVISION = 'dbe34bb2407164f70883c661c76664f0d596e522'
CHUNK = 4 * 1024**2
MAX_SECONDS = 7200
START = time.monotonic()
LOCK = threading.Lock()


def write(path, value):
    temp = path.with_name(path.name + '.pending')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def fetch(asset, lo):
    hi = min(lo + CHUNK, asset['size']) - 1
    errors = []
    for attempt in range(4):
        assert time.monotonic() - START < MAX_SECONDS, 'Input recovery wall-clock limit reached'
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        # API metadata comes from the official HF endpoint; only payload transport
        # uses the accessible public mirror and is checked against official SHA.
        url = 'https://hf-mirror.com/datasets/LidongYang/EEG_Image_decode/resolve/' + REVISION + '/' + asset['path']
        req = urllib.request.Request(url, headers={'Range': f'bytes={lo}-{hi}',
              'Accept-Encoding': 'identity', 'User-Agent': 'EEG-full-data-research/1.0'})
        try:
            with opener.open(req, timeout=12) as response:
                assert response.status == 206
                assert response.headers.get('Content-Range') == f'bytes {lo}-{hi}/{asset["size"]}'
                b = response.read(hi - lo + 2)
            assert len(b) == hi - lo + 1
            return lo, b, dict(start=lo, end=hi, bytes=len(b), sha256=hashlib.sha256(b).hexdigest(), attempts=attempt+1)
        except Exception as exc:
            errors.append(type(exc).__name__)
    raise RuntimeError(f'Range failed: {asset["path"]}, offset={lo}, errors={errors}')


def download(asset, subject, split):
    part = STAGE / f'{subject}_{split}.range.part'
    journal = OUT / f'{subject}_{split}_download_chunks.jsonl'
    if not part.exists():
        assert shutil.disk_usage(OUT).free > asset['size'] + 3 * 1024**3, 'Insufficient server staging headroom'
        with part.open('xb') as f:
            f.truncate(asset['size'])
    assert part.is_file() and not part.is_symlink() and part.stat().st_size == asset['size']
    completed = {}
    if journal.exists():
        with part.open('rb') as f:
            for line in journal.read_text().splitlines():
                row = json.loads(line); f.seek(row['start']); b = f.read(row['bytes'])
                assert hashlib.sha256(b).hexdigest() == row['sha256'], 'Corrupted recovery chunk'
                completed[row['start']] = row
    offsets = [lo for lo in range(0, asset['size'], CHUNK) if lo not in completed]
    fd = os.open(part, os.O_RDWR)
    done = sum(r['bytes'] for r in completed.values())
    try:
        with futures.ThreadPoolExecutor(max_workers=8) as pool, journal.open('a') as log:
            pending = {}; source = iter(offsets)
            def add():
                lo = next(source, None)
                if lo is not None:
                    pending[pool.submit(fetch, asset, lo)] = lo
            for _ in range(16):
                add()
            while pending:
                ready, _ = futures.wait(pending, return_when=futures.FIRST_COMPLETED)
                for job in ready:
                    pending.pop(job)
                    lo, b, row = job.result()
                    assert os.pwrite(fd, b, lo) == len(b)
                    log.write(json.dumps(row) + '\n'); log.flush()
                    completed[lo] = row; done += len(b)
                    add()
                progress = dict(stage='DOWNLOADING', subject=subject, split=split, completed_bytes=done,
                                total_bytes=asset['size'], completed_chunks=len(completed),
                                elapsed_seconds=time.monotonic()-START, pid=os.getpid(), formal_training=False)
                write(OUT / 'input_recovery_progress.json', progress)
            os.fsync(fd)
    finally:
        os.close(fd)
    assert done == asset['size']
    assert sha(part) == asset['lfs']['oid'], 'Whole official LFS hash mismatch'
    return part


def save_tensor(subject, split, repeats, tensor, asset, source_shape, channels, times, trial_indices):
    name = f'{subject}_{split}_mean{repeats}.pt'
    p = OUT / name
    assert tensor.dtype == torch.float32 and torch.isfinite(tensor).all()
    if p.exists():
        assert torch.equal(torch.load(p, weights_only=True, mmap=True), tensor), ('Conflicting existing output', p)
    else:
        torch.save(tensor, p)
    identity = dict(bytes=p.stat().st_size, sha256=sha(p))
    layout = dict(subject=subject, split=split, output=name, output_identity=identity,
                  shape=list(tensor.shape), dtype='float32', raw_shape=source_shape, raw_dtype='float64',
                  physical_repeat_indices=trial_indices, repeat_protocol=f'R{repeats}',
                  channels=channels, times=times,
                  source_asset=asset['path'], source_identity=dict(asset=asset['path'], bytes=asset['size'],
                       sha256=asset['lfs']['oid'], author_whole_SHA_verified=True),
                  preprocessing='author MVNN; cast original trials to float32, then mean selected physical repeats; no new normalization',
                  independent_event_label_replay=False)
    write(OUT / (p.stem + '.layout.json'), layout)
    return dict(path=str(p), **identity)


def derive(asset, subject, split):
    ready = OUT / f'{subject}_{split}_ready.json'
    if ready.exists():
        receipt = json.loads(ready.read_text())
        assert receipt['source_sha256'] == asset['lfs']['oid']
        for output in receipt['outputs']:
            p = Path(output['path']); assert p.exists() and sha(p) == output['sha256']
        return receipt
    part = download(asset, subject, split)
    # NumPy's loader handles the author's protocol-4 pickle despite the .npy suffix.
    data = np.load(part, allow_pickle=True)
    assert isinstance(data, dict) and set(data) == {'preprocessed_eeg_data', 'ch_names', 'times'}
    raw = data['preprocessed_eeg_data']; shape = [16540, 4, 63, 250] if split == 'train' else [200, 80, 63, 250]
    assert list(raw.shape) == shape and raw.dtype == np.float64
    channels = list(data['ch_names']); times = np.asarray(data['times'])[50:]
    assert len(channels) == 63 and times.shape == (250,)
    assert np.allclose(times, np.arange(250)/250, atol=1e-12)
    outputs = []
    if split == 'train':
        compact = torch.empty((16540, 63, 250), dtype=torch.float32)
        for i in range(0, 16540, 64):
            block = np.asarray(raw[i:i+64], dtype=np.float32)
            assert np.isfinite(block).all()
            compact[i:i+len(block)] = torch.from_numpy(block.mean(1))
        if subject == 'sub-02':
            previous = Path('/root/mced_eeg_full_montage_cache_v3')
            layout = json.loads((previous / 'sub-02_train_layout.json').read_text())
            old = torch.load(previous / 'sub-02_calibration_mean4.pt', weights_only=True)
            assert torch.equal(compact[layout['selected_image_indices']], old)
        outputs.append(save_tensor(subject, split, 4, compact, asset, shape, channels, times.tolist(), [0,1,2,3]))
        del compact
    else:
        # These fixed physical repeat indices were registered before any new evaluation.
        first4 = np.asarray(raw[:, :4], dtype=np.float32)
        assert np.isfinite(raw).all()
        pair_differences = [float(np.abs(first4[:,j]-first4[:,0]).mean()) for j in [1,2,3]]
        assert min(pair_differences) > 0, 'Repeated trial copies rather than physical trials'
        physical = OUT / f'{subject}_test_first4_physical.pt'
        if physical.exists():
            assert torch.equal(torch.load(physical, weights_only=True, mmap=True), torch.from_numpy(first4))
        else:
            torch.save(torch.from_numpy(first4), physical)
        outputs.append(dict(path=str(physical), bytes=physical.stat().st_size, sha256=sha(physical)))
        outputs.append(save_tensor(subject, split, 1, torch.from_numpy(first4[:,0].copy()), asset, shape, channels, times.tolist(), [0]))
        outputs.append(save_tensor(subject, split, 4, torch.from_numpy(first4.mean(1)), asset, shape, channels, times.tolist(), [0,1,2,3]))
        mean80 = np.empty((200, 63, 250), dtype=np.float32)
        for i in range(0, 200, 16):
            mean80[i:i+16] = np.asarray(raw[i:i+16], dtype=np.float32).mean(1)
        previous = Path('/root/mced_eeg_full_montage_cache_v3') / f'{subject}_test_mean80.pt'
        if previous.exists():
            assert torch.equal(torch.from_numpy(mean80), torch.load(previous, weights_only=True))
        outputs.append(save_tensor(subject, split, 80, torch.from_numpy(mean80), asset, shape, channels, times.tolist(), list(range(80))))
        del first4, mean80
    del raw, data
    receipt = dict(status='VERIFIED_INPUTS_READY', subject=subject, split=split,
                   source_asset=asset['path'], source_sha256=asset['lfs']['oid'], whole_LFS_SHA_verified=True,
                   outputs=outputs, time_alignment='EEG unchanged; metadata times[50:] only',
                   generation_or_metric_calls=0, formal_training=False,
                   elapsed_seconds=time.monotonic()-START)
    write(ready, receipt)
    # Delete only this newly created staging file after durable derivative receipts.
    assert part.resolve().parent == STAGE.resolve() and not part.is_symlink()
    part.unlink()
    print(json.dumps(receipt), flush=True)
    return receipt


def main():
    OUT.mkdir(parents=True, exist_ok=True); STAGE.mkdir(exist_ok=True)
    lock = (OUT / 'input_recovery.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(2)
    tree = json.loads((ROOT / 'audit/full_author_asset_tree.json').read_text())
    files = {r['path']: r for r in tree if r['type'] == 'file'}
    wanted = [('sub-02', 'train')] + [(f'sub-{i:02d}', 'test') for i in range(1,11)]
    tasks = []
    for subject, split in wanted:
        asset = files[f'Preprocessed_data_250Hz/{subject}/preprocessed_eeg_{"training" if split == "train" else "test"}.npy']
        assert asset['size'] == (8336163007 if split == 'train' else 2016003002)
        assert asset['lfs']['size'] == asset['size'] and len(asset['lfs']['oid']) == 64
        tasks.append((asset, subject, split))
    write(OUT / 'input_recovery_plan.json', dict(pid=os.getpid(), revision=REVISION,
          assets=[dict(path=a['path'], bytes=a['size'], sha256=a['lfs']['oid']) for a,_,_ in tasks],
          source_metadata='official HF API, pinned immutable revision; accessible payload mirror verified with official full SHA',
          fixed_test_physical_repeat_indices=[0,1,2,3], max_wall_seconds=MAX_SECONDS,
          source_staging_policy='one asset at a time; remove owned source only after all derivatives are hashed and receipted',
          no_local_C_D_large_files=True, formal_training=False))
    try:
        results = [derive(a,s,sp) for a,s,sp in tasks]
        write(OUT / 'input_recovery_completion.json', dict(status='COMPLETED', assets=len(results),
              elapsed_seconds=time.monotonic()-START, formal_training=False))
    except Exception as exc:
        write(OUT / 'input_recovery_failure.json', dict(status='FAILED_RESUMABLE', type=type(exc).__name__,
              message=str(exc), traceback=traceback.format_exc(), elapsed_seconds=time.monotonic()-START))
        raise


if __name__ == '__main__':
    main()
