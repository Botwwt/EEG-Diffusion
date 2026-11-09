"""Refresh server storage and dependency evidence without deleting any files.

Old inventories are not deletion manifests. This entry deliberately has no apply
mode: its candidates must be checked against current results and runtime uses.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time

ROOT = Path(__file__).resolve().parent
WORK = Path('/root/brainhive_reproduction_20261020_v1')
DATA = Path('/root/autodl-tmp')
UNRELATED_RUNS = tuple(DATA / 'mced_research_bundle/runs' / name for name in (
    'cifar10_face_ann_pilot_20261011', 'cifar10_face_snn_pilot_20261011',
    'cifar10_velocity_ann_pilot_20261011', 'cifar10_velocity_snn_pilot_20261011'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_new(path, value):
    payload = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    assert len(payload) < 8 << 20, 'Keep this audit bounded; do not truncate evidence'
    with path.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def current_dependencies():
    """Read the literal current import-path definition without importing Torch."""
    import ast
    tree = ast.parse((ROOT / 'brainhive_native_models_v1.py').read_text())
    matches = [node.value for node in tree.body if isinstance(node, ast.Assign)
               and any(isinstance(x, ast.Name) and x.id == 'PYTHON_PATHS' for x in node.targets)]
    assert len(matches) == 1
    paths = ast.literal_eval(matches[0])
    assert all(isinstance(x, str) and x.startswith('/root/') for x in paths)
    return paths


def process_references():
    result = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            arguments = (proc / 'cmdline').read_bytes().split(b'\0')
            command = Path(arguments[0].decode(errors='replace')).name if arguments[0] else ''
        except OSError:
            continue
        opened = set()
        links = [proc / 'exe', proc / 'cwd']
        try:
            links.extend((proc / 'fd').iterdir())
        except OSError:
            pass
        for link in links:
            try:
                value = os.readlink(link).removesuffix(' (deleted)')
                if value.startswith(('/root/', '/dev/shm/')):
                    opened.add(value)
            except OSError:
                pass
        try:
            for line in (proc / 'maps').read_text().splitlines():
                parts = line.split(None, 5)
                if len(parts) == 6 and parts[5].startswith(('/root/', '/dev/shm/')):
                    opened.add(parts[5].removesuffix(' (deleted)'))
        except OSError:
            pass
        if command or opened:
            result.append(dict(pid=int(proc.name), command=command, opened_or_mapped=sorted(opened)))
    return result


def inventory(started):
    groups = defaultdict(lambda: dict(files=0, logical_bytes=0, unique_allocated_bytes=0))
    seen = set()
    large = []
    errors = []
    bytecode = dict(files=0, allocated_bytes=0)
    root_dev = Path('/root').stat().st_dev
    data_dev = DATA.stat().st_dev
    for directory, dirs, files in os.walk('/root', followlinks=False,
                                           onerror=lambda e: errors.append(str(e))):
        assert time.monotonic() - started < 600, 'Incomplete inventory must fail'
        parent = Path(directory)
        dirs[:] = [name for name in dirs if not (parent / name).is_symlink()]
        for name in files:
            path = parent / name
            try:
                details = path.lstat()
                if not stat.S_ISREG(details.st_mode):
                    continue
                if details.st_dev not in {root_dev, data_dev}:
                    continue
                if path.is_relative_to(DATA):
                    group = DATA / path.relative_to(DATA).parts[0]
                else:
                    group = Path('/root') / path.relative_to('/root').parts[0]
                row = groups[str(group)]
                row['files'] += 1
                row['logical_bytes'] += details.st_size
                identity = (details.st_dev, details.st_ino)
                first_inode = identity not in seen
                seen.add(identity)
                allocated = details.st_blocks * 512
                if first_inode:
                    row['unique_allocated_bytes'] += allocated
                if details.st_size >= 32 << 20:
                    large.append(dict(path=str(path), bytes=details.st_size,
                        allocated_bytes=allocated if first_inode else 0,
                        dev=details.st_dev, inode=details.st_ino, links=details.st_nlink,
                        mtime_ns=details.st_mtime_ns,
                        classification='UNREVIEWED_PRESERVE'))
                if name.endswith('.pyc') and '__pycache__' in path.parts:
                    source = path.parent.parent / (name.split('.')[0] + '.py')
                    if source.is_file():
                        bytecode['files'] += 1
                        bytecode['allocated_bytes'] += allocated if first_inode else 0
            except OSError as exc:
                errors.append(str(exc))
    return dict(top_level=groups, large_files=sorted(large, key=lambda x: -x['bytes']),
                bytecode_with_existing_source=bytecode, read_errors=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert os.name == 'posix' and ROOT == Path(
        '/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1')
    out = Path(args.output)
    assert out.parent == ROOT / 'audit' and out.name.endswith('.json') and not out.exists()
    started = time.monotonic()
    recipe = json.loads((ROOT / 'configs/functional_profiles_20261022_v1/O.json').read_text())
    for name, digest in recipe['source_sha256'].items():
        assert sha(ROOT / name) == digest, ('Pinned source changed', name)
    before = {str(path): sha(path) for path in [ROOT / 'results.tsv',
        ROOT / 'configs/BrainHIVE_bounded_main_recipe_v1.json',
        *[ROOT / name for name in recipe['source_sha256']]]}
    inputs = [dict(path=name, exists=Path(name).is_file(), expected_sha256=digest)
              for name, digest in recipe['input_sha256'].items()]
    dependencies = current_dependencies()
    report = dict(status='READ_ONLY_STORAGE_AUDIT_NO_CLEANUP_ADMISSION',
        UTC=datetime.now(timezone.utc).isoformat(), source_sha256=sha(__file__),
        free={name: shutil.disk_usage(name)._asdict() for name in ['/root', str(DATA), '/dev/shm']},
        current_dependency_paths=dependencies, profile_input_presence=inputs,
        presence_is_not_a_full_input_SHA_audit=True,
        processes=process_references(), inventory=inventory(started),
        unrelated_CIFAR_run_directories_to_review=[str(p) for p in UNRELATED_RUNS if p.is_dir()],
        retained_result_metadata_copy=str(ROOT / 'audit/retained_result_metadata_20261022_v1.zip'),
        final_weights_result_images_EEG_and_archives_must_be_preserved=True,
        deleted_files=0, GPU_calls=0, formal_training_started=False)
    report['protected_before_sha256'] = before
    assert {name: sha(name) for name in before} == before
    report['seconds'] = time.monotonic() - started
    write_new(out, report)
    print(json.dumps(dict(status=report['status'], receipt=str(out),
        large_files=len(report['inventory']['large_files']),
        read_errors=len(report['inventory']['read_errors']), deleted_files=0)))


if __name__ == '__main__':
    main()
