"""Retire one unused AVDE virtual environment, preserving its rebuild inventory.

No dataset, model checkpoint, experiment image, or project source is a deletion
target. The target is fixed rather than accepted from an arbitrary command line.
"""
from pathlib import Path
import csv
import datetime
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent
TARGET = Path('/root/autodl-tmp/avde_repro_env_20261015')
PROOF = ROOT/'audit/unused_AVDE_environment_retirement_20261020_v1.json'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    start = time.monotonic()
    assert not PROOF.exists(), 'Preserve the previous action receipt'
    assert TARGET == TARGET.resolve() and not TARGET.is_symlink()
    assert TARGET.parent == Path('/root/autodl-tmp').resolve()
    assert (TARGET/'pyvenv.cfg').is_file()
    assert not Path(sys.executable).resolve().is_relative_to(TARGET)
    cfg = (TARGET/'pyvenv.cfg').read_text()
    assert 'include-system-site-packages = true' in cfg
    dependency_path = ROOT/'audit/obsolete_AVDE_environment_dependency_check_20261020_v1.json'
    dependency = json.loads(dependency_path.read_text())
    assert dependency['resolved'] == str(TARGET)
    assert all('file' in row and '/audit/' in row['file'] for row in dependency['current_references'])

    # Recheck live processes and actual mappings immediately before deletion.
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        for name in ['cmdline', 'maps']:
            try:
                body = (p/name).read_bytes()
            except (FileNotFoundError, ProcessLookupError):
                continue
            assert str(TARGET).encode() not in body, ('Live environment dependency', p.name, name)
    for base in [ROOT, Path('/root/mced_representation_runtime_v1')]:
        for p in base.rglob('*'):
            if p.is_symlink():
                assert not p.resolve().is_relative_to(TARGET), ('Dependent symlink', str(p))
            if base == ROOT and p.suffix in ['.py', '.sh'] and p.is_file() and p != Path(__file__):
                if any(x in p.parts for x in ['.git', 'reproduction_sources']):
                    continue
                assert str(TARGET) not in p.read_text(errors='replace'), ('Executable dependency', str(p))

    protected = [
        Path('/root/autodl-tmp/legacy_eeg_mean4_archive_20261016')/f'sub-{s:02}_train_mean4.pt'
        for s in [8, 9, 10]
    ] + [
        ROOT/'FINAL_REPORT.md', ROOT/'results.tsv',
        ROOT/'configs/compute_time_bounds.json',
        ROOT/'audit/idea_completed_three_arm_saved_summaries_20261020_v1.json',
    ]
    before = {str(p): (p.stat().st_dev, p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns)
              for p in protected}
    package_dir = TARGET/'lib/python3.12/site-packages'
    distributions = sorted(
        [{'name': d.metadata.get('Name'), 'version': d.version}
         for d in importlib.metadata.distributions(path=[str(package_dir)])],
        key=lambda row: (row['name'] or '', row['version']))
    inventory = ROOT/'audit/unused_AVDE_environment_files_20261020_v1.csv'
    files = logical_bytes = exclusive_blocks = 0
    with inventory.open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['path', 'kind', 'bytes', 'sha256', 'symlink_target', 'links'])
        w.writeheader()
        for p in sorted(TARGET.rglob('*')):
            if p.is_symlink():
                w.writerow(dict(path=str(p.relative_to(TARGET)), kind='symlink', symlink_target=os.readlink(p)))
            elif p.is_file():
                stat = p.stat()
                w.writerow(dict(path=str(p.relative_to(TARGET)), kind='file', bytes=stat.st_size,
                                sha256=sha(p), links=stat.st_nlink))
                files += 1
                logical_bytes += stat.st_size
                if stat.st_nlink == 1:
                    exclusive_blocks += stat.st_blocks*512
    free_before = shutil.disk_usage(TARGET.parent).free
    receipt = dict(status='VERIFIED_UNUSED_ENVIRONMENT_READY_FOR_RETIREMENT',
                   target=str(TARGET), pyvenv_cfg=cfg, distributions=distributions,
                   file_inventory=str(inventory), file_inventory_sha256=sha(inventory),
                   files=files, logical_bytes=logical_bytes, exclusive_file_allocated_bytes=exclusive_blocks,
                   free_bytes_before=free_before, dependency_check_sha256=sha(dependency_path),
                   source_sha256=sha(__file__), GPU_used=False,
                   datasets_deleted=0, checkpoints_deleted=0, generated_images_deleted=0,
                   project_source_files_deleted=0,
                   UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    PROOF.write_text(json.dumps(receipt, indent=2)+'\n')
    # shutil.rmtree unlinks directory symlinks; it does not recurse into lib64.
    assert TARGET.resolve() == Path('/root/autodl-tmp/avde_repro_env_20261015')
    shutil.rmtree(TARGET)
    assert not TARGET.exists()
    for p in protected:
        stat = p.stat()
        assert before[str(p)] == (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    receipt.update(status='RETIRED_UNUSED_AVDE_ENVIRONMENT_PROTECTED_ARTIFACTS_UNCHANGED',
                   free_bytes_after=shutil.disk_usage(TARGET.parent).free,
                   actual_free_bytes_increase=shutil.disk_usage(TARGET.parent).free-free_before,
                   protected_artifacts_stat_unchanged=list(before), seconds=time.monotonic()-start)
    PROOF.write_text(json.dumps(receipt, indent=2)+'\n')
    with (ROOT/'audit/CPU_postprocessing_ledger.jsonl').open('a') as f:
        f.write(json.dumps(dict(stage='retire_unused_AVDE_environment', GPU_used=False,
                               seconds=receipt['seconds'], receipt=str(PROOF), receipt_sha256=sha(PROOF)))+'\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'distributions'}), flush=True)


if __name__ == '__main__':
    main()
