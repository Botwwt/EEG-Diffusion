"""Exact cleanup of unrelated CIFAR/MNIST models and reproducible runtime caches.

EEG models, generated images, metrics, source files, logs and exact old archives
are excluded. No recursive directory deletion or environment removal is used.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import time

from cleanup_BrainHIVE_baseline_preserving_CPU_20261022_v1 import live_uses

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'audit/expanded_disposable_cleanup_20261022_v1'
BUNDLE = Path('/root/autodl-tmp/mced_research_bundle')
CIFAR_NAMES = ('cifar10_face_ann_pilot_20261011', 'cifar10_face_snn_pilot_20261011',
               'cifar10_velocity_ann_pilot_20261011', 'cifar10_velocity_snn_pilot_20261011')
MNIST_RUN = BUNDLE / 'studies/snn_velocity_low_noise_v1/run'
MNIST_RAW = Path('/root/autodl-tmp/mced_bootstrap_20261011/original_snapshot/data/MNIST/raw')
RAW_NAMES = {split + '-' + kind for split in ('train', 't10k')
             for kind in ('images-idx3-ubyte', 'labels-idx1-ubyte')}
RAW_NAMES |= {name + '.gz' for name in tuple(RAW_NAMES)}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def identity(path):
    assert path.is_absolute() and path.resolve(strict=True) == path and not path.is_symlink()
    s = path.lstat()
    assert stat.S_ISREG(s.st_mode)
    return [s.st_dev, s.st_ino, s.st_size, s.st_nlink, s.st_mtime_ns]


def classify(path):
    if path.parent.parent == BUNDLE / 'runs' and path.parent.name in CIFAR_NAMES:
        if path.name in {'latest.pt', 'step_002000.pt', 'step_020000.pt'}:
            return 'UNRELATED_CIFAR_MODEL'
    if path.parent == MNIST_RUN and path.name in {'initialization.pt', 'latest.pt', 'step_002000.pt', 'step_006000.pt'}:
        return 'UNRELATED_MNIST_MODEL'
    if path.parent == MNIST_RAW and path.name in RAW_NAMES:
        return 'UNRELATED_PUBLIC_MNIST_DATA'
    if path.parent == Path('/root/miniconda3/pkgs'):
        suffix = next((s for s in ('.conda', '.tar.bz2') if path.name.endswith(s)), None)
        if suffix:
            installed = Path('/root/miniconda3/conda-meta') / (path.name[:-len(suffix)] + '.json')
            if installed.is_file():
                return 'REACQUIRABLE_INSTALLED_PACKAGE_ARCHIVE'
    if path.is_relative_to('/root') and path.parent.name == '__pycache__' and path.suffix == '.pyc':
        source = path.parent.parent / (path.name.split('.')[0] + '.py')
        if source.is_file():
            return 'REBUILDABLE_PYTHON_BYTECODE_WITH_SOURCE'
    return None


def snapshot():
    return {name: shutil.disk_usage(name)._asdict() for name in ('/root', '/root/autodl-tmp')}


def self_check():
    assert classify(BUNDLE / 'runs' / CIFAR_NAMES[0] / 'latest.pt')
    assert classify(MNIST_RAW / 'train-images-idx3-ubyte')
    for path in [BUNDLE / 'runs/idea_first_ATMS_patience50_v1/latest.pt',
                 BUNDLE / 'runs' / CIFAR_NAMES[0] / 'sample.png',
                 BUNDLE / 'runs' / CIFAR_NAMES[0] / 'config.json',
                 MNIST_RUN / 'results.csv', MNIST_RUN / 'checkpoint.pt.bak',
                 Path('/root/brainhive_reproduction_20261020_v1/components/bounded_v1/risk/component_final.pt'),
                 Path('/root/autodl-tmp/retained_legacy_exact_archives_20261022_v1/AVDE_preprocessed_eeg_training.npy.EXACT_BS4_ZLIB_v1')]:
        assert classify(path) is None, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    self_check()
    if args.self_check:
        print('UNRELATED_MODEL_SCOPE_PROTECTS_EEG_MODELS_IMAGES_RESULTS_AND_ARCHIVES'); return
    assert ROOT == Path('/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1')
    assert not BASE.exists()
    started = time.monotonic()
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    for name in CIFAR_NAMES:
        config = json.loads((BUNDLE / 'runs' / name / 'config.json').read_text())
        assert config['dataset'] == 'cifar10' and config['size'] == 32 and config['channels'] == 3
    plan = json.loads((MNIST_RUN.parent / 'plan.json').read_text())
    assert plan['config']['dataset'] == 'mnist' and plan['config']['channels'] == 1
    recipe = json.loads((ROOT / 'configs/functional_profiles_20261022_v1/O.json').read_text())
    protected = {str(ROOT / name): digest for name, digest in recipe['source_sha256'].items()}
    protected[str(ROOT / 'results.tsv')] = sha(ROOT / 'results.tsv')
    for name in CIFAR_NAMES:
        for path in (BUNDLE / 'runs' / name).rglob('*'):
            if path.is_file() and not classify(path):
                protected[str(path)] = sha(path)
    for path in MNIST_RUN.parent.rglob('*'):
        if path.is_file() and not classify(path):
            protected[str(path)] = sha(path)
    targets = []
    for parent, dirs, files in os.walk('/root', followlinks=False):
        assert time.monotonic() - started < 900
        parent = Path(parent)
        dirs[:] = [name for name in dirs if not (parent / name).is_symlink() and name != '.git']
        for name in files:
            path = parent / name
            kind = classify(path)
            if not kind or path.is_symlink() or path.resolve(strict=True) != path:
                continue
            id_ = identity(path)
            if id_[3] != 1:
                continue  # Do not overstate reclaim from shared installation inodes.
            assert str(path) not in protected and str(path) not in recipe['input_sha256']
            row = dict(path=str(path), kind=kind, identity=id_, allocated_bytes=path.stat().st_blocks * 512,
                       sha256=sha(path))
            if kind == 'REBUILDABLE_PYTHON_BYTECODE_WITH_SOURCE':
                source = path.parent.parent / (name.split('.')[0] + '.py')
                row.update(source=str(source), source_sha256=sha(source))
            targets.append(row)
    assert sum(t['kind'] == 'UNRELATED_CIFAR_MODEL' for t in targets) == 12
    assert not live_uses([Path(x['path']) for x in targets])
    for name, digest in protected.items():
        assert sha(name) == digest, name
    BASE.mkdir()
    before = snapshot()
    dump(BASE / 'plan.json', dict(status='EXACT_SCOPE_REVIEWED', before=before, targets=targets,
        protected_sha256=protected, counts=dict(Counter(x['kind'] for x in targets)),
        bytes=sum(x['allocated_bytes'] for x in targets), source_sha256=sha(__file__),
        EEG_models_images_metrics_logs_and_sources_deleted=0,
        unrelated_CIFAR_MNIST_model_payloads_will_not_be_retained=True))
    if args.apply:
        with (BASE / 'deletions.jsonl').open('x') as journal:
            for row in targets:
                path = Path(row['path'])
                assert classify(path) == row['kind'] and identity(path) == row['identity']
                assert sha(path) == row['sha256']
                if 'source' in row:
                    assert sha(row['source']) == row['source_sha256']
                path.unlink()
                journal.write(json.dumps(row) + '\n'); journal.flush()
            os.fsync(journal.fileno())
        for name, digest in protected.items():
            assert sha(name) == digest, name
        after = snapshot()
        receipt = dict(status='EXACT_UNRELATED_MODEL_AND_REBUILDABLE_CACHE_CLEANUP_COMPLETE',
            UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), before=before, after=after,
            counts=dict(Counter(x['kind'] for x in targets)), deleted_files=len(targets),
            deleted_allocated_bytes=sum(x['allocated_bytes'] for x in targets),
            actual_net_free_change_bytes=sum(after[p]['free'] - before[p]['free'] for p in before),
            all_protected_payload_SHA_unchanged=True, source_sha256=sha(__file__),
            EEG_models_images_metrics_logs_and_sources_deleted=0, GPU_calls=0,
            formal_training_started=False, seconds=time.monotonic() - started)
        dump(BASE / 'receipt.json', receipt)
        print(json.dumps(receipt))
    else:
        print(json.dumps(dict(status='PREPARED_NO_DELETION', targets=len(targets))))


if __name__ == '__main__':
    main()
