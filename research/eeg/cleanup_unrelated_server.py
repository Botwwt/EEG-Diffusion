"""User-authorized server-only cleanup, with explicit paths and durable receipts."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import time

ROOT = Path('/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1')
AUDIT = ROOT / 'audit/storage_cleanup_20261018_v1'
BASE = Path('/root/autodl-tmp/mced_research_bundle')
PROTECTED = [
    Path('/root/atm_diffusion_20261015'),
    Path('/root/mced_eeg_full_montage_cache_v3'),
    Path('/root/mced_eeg_single_visual_persistent_checkpoints_v1'),
    Path('/root/autodl-tmp/legacy_eeg_mean4_archive_20261016'),
    Path('/root/autodl-tmp/atm_published_training_trials_20261016_v2'),
    BASE / 'external/things_eeg2_direct_v1',
    BASE / 'studies/eeg_clip_visual_feature_cache_v2',
    Path('/dev/shm/eeg_diffusion_next_20261018_v1'),
]


def dump(name, obj):
    (AUDIT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def disk():
    return {p: dict(zip(('total', 'used', 'free'), shutil.disk_usage(p)))
            for p in ['/', '/root/autodl-tmp', '/dev/shm']}


def validate(p):
    assert p.is_absolute() and not p.is_symlink(), str(p)
    assert p.resolve() == p, str(p)
    assert p.is_relative_to('/root/autodl-tmp'), str(p)
    assert p != Path('/root/autodl-tmp') and p != BASE, str(p)
    for keep in PROTECTED:
        assert not (p == keep or p.is_relative_to(keep) or keep.is_relative_to(p)), (p, keep)


def live_uses(targets):
    hits = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        links = [proc / 'cwd', proc / 'exe']
        try:
            links.extend((proc / 'fd').iterdir())
        except OSError:
            pass
        values = []
        for link in links:
            try:
                values.append(os.readlink(link).removesuffix(' (deleted)'))
            except OSError:
                pass
        try:
            values.extend(line.split(None, 5)[-1] for line in (proc / 'maps').read_text().splitlines()
                          if len(line.split(None, 5)) == 6)
        except OSError:
            pass
        for value in values:
            for target in targets:
                if value == str(target) or value.startswith(str(target) + '/'):
                    hits.append(dict(pid=int(proc.name), target=str(target), opened=value))
    return hits


def plan():
    AUDIT.mkdir(parents=True, exist_ok=False)
    prefixes = ('cifar', 'mnist', 'prior_mnist', 'imagenet_', 'edm_primary_',
                'dit_conversion_', 'dit_official_', 'spikingdit_', 'sddpm_')
    targets = [(p, 'Unrelated MNIST/CIFAR/ImageNet image diffusion experiment')
               for p in sorted((BASE / 'studies').iterdir())
               if p.is_dir() and p.name.startswith(prefixes)]
    external_names = [
        'adm_metric_v1', 'adm_metric_v2', 'adm_reference_stats_v1',
        'dit_official_assets_v1', 'dit_official_reference_v1', 'edm_primary_baseline_v1',
        'imagenet_autodl_v1', 'spikingdit_extension_v1', 'spikingdit_reference_v1',
        'spikingdit_reference_v2', 'spikingdit_reference_v3', 'spikingdit_reference_v4',
        'spikingdit_runtime_v1',
    ]
    targets.extend((BASE / 'external' / name, 'Unrelated image diffusion assets')
                   for name in external_names if (BASE / 'external' / name).exists())
    targets.extend([
        (BASE / 'data', 'Only MNIST and CIFAR10, explicitly inventoried'),
        (BASE / 'baselines', 'Only SDDPM CIFAR10 and SpikingJelly, explicitly inventoried'),
        (Path('/root/autodl-tmp/icd_eeg_20261017'),
         'BCI-IV2a motor imagery decoder experiments; unrelated to visual EEG reconstruction'),
    ])
    paths = [p for p, _ in targets]
    for p in paths:
        validate(p)
    assert not any(a != b and a.is_relative_to(b) for a in paths for b in paths)
    uses = live_uses(paths)
    dump('live_use_check.json', uses)
    assert not uses, uses
    rows = []
    with (AUDIT / 'file_inventory.jsonl').open('w') as inventory:
        for p, reason in targets:
            logical = allocated = count = 0
            for current, dirs, files in os.walk(p, followlinks=False):
                for name in files:
                    f = Path(current) / name
                    s = f.lstat()
                    item = dict(path=str(f), bytes=s.st_size, blocks=s.st_blocks,
                                mtime_ns=s.st_mtime_ns, symlink=f.is_symlink())
                    if f.is_symlink():
                        item['link'] = os.readlink(f)
                    inventory.write(json.dumps(item, ensure_ascii=False) + '\n')
                    count += 1; logical += s.st_size; allocated += s.st_blocks * 512
            rows.append(dict(path=str(p), reason=reason, files=count,
                             logical_bytes=logical, allocated_bytes=allocated))
    references = []
    for source_root in [ROOT, ROOT.parent / 'eeg_diffusion_next_20261018_v1']:
        for f in source_root.rglob('*'):
            if not f.is_file() or f.suffix not in ('.py', '.yaml', '.yml') or f == Path(__file__):
                continue
            for i, line in enumerate(f.read_text(errors='replace').splitlines(), 1):
                if any(str(p) in line for p in paths):
                    references.append(dict(file=str(f), line=i, text=line[:500]))
    dump('source_references.json', references)
    assert not references, references
    result = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  authorization='User: only retain data and code required for current EEG reconstruction idea; clean server; no large files to C/D.',
                  protected=[str(p) for p in PROTECTED], before=disk(), targets=rows,
                  allocated_bytes=sum(r['allocated_bytes'] for r in rows),
                  files=sum(r['files'] for r in rows), live_uses=uses)
    dump('plan.json', result)
    print(json.dumps(dict(status='PLAN_READY', directories=len(rows), files=result['files'],
                          allocated_GiB=result['allocated_bytes'] / 2**30,
                          before=result['before']), indent=2))


def apply():
    assert not (AUDIT / 'receipt.json').exists()
    data = json.loads((AUDIT / 'plan.json').read_text())
    paths = [Path(row['path']) for row in data['targets']]
    for p in paths:
        validate(p)
    assert not live_uses(paths), 'A live process now uses a cleanup target'
    # Preserve small source/config/report records on the server before payload removal.
    saved_bytes = saved_files = 0
    with tarfile.open(AUDIT / 'small_sources_reports.tar.gz', 'x:gz') as archive:
        for p in paths:
            for current, dirs, files in os.walk(p, followlinks=False):
                dirs[:] = [d for d in dirs if d not in ('.git', '__pycache__', 'vendor', '.venv')]
                for name in files:
                    f = Path(current) / name
                    if f.is_symlink() or f.suffix not in ('.py', '.sh', '.md', '.yaml', '.yml', '.json', '.csv', '.tsv', '.txt', '.toml'):
                        continue
                    if f.stat().st_size > 1024**2 or saved_bytes + f.stat().st_size > 50 * 1024**2:
                        continue
                    archive.add(f, arcname=str(f).removeprefix('/'), recursive=False)
                    saved_bytes += f.stat().st_size; saved_files += 1
    removed = []
    with (AUDIT / 'deletions.jsonl').open('x') as log:
        for p in paths:
            validate(p)
            shutil.rmtree(p)
            assert not p.exists()
            row = dict(path=str(p), status='REMOVED', utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
            log.write(json.dumps(row) + '\n'); log.flush(); os.fsync(log.fileno())
            removed.append(row)
    retained = {str(p): p.exists() for p in PROTECTED}
    assert all(retained.values()), retained
    after = disk()
    result = dict(status='COMPLETED', before=data['before'], after=after,
                  removed_directories=len(removed), retained=retained,
                  freed_bytes={p: after[p]['free']-data['before'][p]['free'] for p in after},
                  archived_small_files=saved_files, archived_small_bytes=saved_bytes,
                  local_large_files_created=0)
    dump('receipt.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['plan', 'apply'])
    args = parser.parse_args()
    plan() if args.mode == 'plan' else apply()
