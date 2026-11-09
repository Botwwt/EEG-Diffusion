"""Remove remaining unrelated class-conditioned image backbone experiments."""
from pathlib import Path
import json
import os
import shutil
import tarfile
import time
from cleanup_unrelated_server import BASE, ROOT, PROTECTED, validate, live_uses, disk

AUDIT = ROOT / 'audit/storage_cleanup_other_images_20261018_v1'
AUDIT.mkdir(parents=True, exist_ok=False)
# Keep all EEG/brain experiments, including EEG probes that used REPA.
KEEP_REPA = {'repa_real_eeg_derivative_preflight_v1', 'repa_real_eeg_feature_probe_v1'}
prefixes = ('repa_', 'raev2_', 'ddim_endpoint_', 'endpoint_inception_', 'endpoint_low_noise_',
            'sdm_', 'terminal_compatibility_')
targets = [p for p in sorted((BASE / 'studies').iterdir())
           if p.is_dir() and p.name.startswith(prefixes) and p.name not in KEEP_REPA]
targets += [p for p in sorted((BASE / 'external').iterdir())
            if p.is_dir() and p.name.startswith('repa_')]
for p in targets:
    validate(p)
assert not live_uses(targets)
assert not any(a != b and a.is_relative_to(b) for a in targets for b in targets)
references = []
for source_root in [ROOT, ROOT.parent / 'eeg_diffusion_next_20261018_v1']:
    for f in source_root.rglob('*'):
        if not f.is_file() or f.suffix not in ('.py', '.yaml', '.yml') or f == Path(__file__):
            continue
        for i, line in enumerate(f.read_text(errors='replace').splitlines(), 1):
            if any(str(p) in line for p in targets):
                references.append(dict(file=str(f), line=i, text=line[:500]))
(AUDIT / 'source_references.json').write_text(json.dumps(references, indent=2))
assert not references, references
before = disk()
rows = []
with (AUDIT / 'file_inventory.jsonl').open('w') as inventory:
    for p in targets:
        allocated = count = 0
        for current, dirs, files in os.walk(p, followlinks=False):
            for name in files:
                f = Path(current) / name
                s = f.lstat(); allocated += s.st_blocks * 512; count += 1
                inventory.write(json.dumps(dict(path=str(f), bytes=s.st_size,
                                                mtime_ns=s.st_mtime_ns, symlink=f.is_symlink())) + '\n')
        rows.append(dict(path=str(p), allocated_bytes=allocated, files=count))
plan = dict(authorization='User: retain only current EEG idea data/code; server only cleanup',
            reason='Unrelated REPA/RAE class-conditioned image experiments, DDIM endpoints and Inception; keep actual EEG probes',
            before=before, targets=rows, keep_eeg_probes=sorted(KEEP_REPA))
(AUDIT / 'plan.json').write_text(json.dumps(plan, indent=2))
saved_bytes = saved_files = 0
with tarfile.open(AUDIT / 'small_sources_reports.tar.gz', 'x:gz') as archive:
    for p in targets:
        for current, dirs, files in os.walk(p, followlinks=False):
            dirs[:] = [d for d in dirs if d not in ('.git', '__pycache__', 'vendor', '.venv')]
            for name in files:
                f = Path(current) / name
                if f.is_symlink() or f.suffix not in ('.py', '.sh', '.md', '.yaml', '.yml', '.json', '.csv', '.tsv', '.txt', '.toml'):
                    continue
                if f.stat().st_size > 1024**2 or saved_bytes + f.stat().st_size > 20 * 1024**2:
                    continue
                archive.add(f, arcname=str(f).removeprefix('/'), recursive=False)
                saved_bytes += f.stat().st_size; saved_files += 1
assert not live_uses(targets)
with (AUDIT / 'deletions.jsonl').open('x') as log:
    for p in targets:
        validate(p)
        shutil.rmtree(p)
        assert not p.exists()
        log.write(json.dumps(dict(path=str(p), status='REMOVED')) + '\n'); log.flush(); os.fsync(log.fileno())
retained = {str(p): p.exists() for p in PROTECTED}
retained.update({str(BASE / 'studies' / n): (BASE / 'studies' / n).exists() for n in KEEP_REPA})
assert all(retained.values()), retained
after = disk()
receipt = dict(status='COMPLETED', removed_directories=len(targets), before=before, after=after,
               freed_bytes={p: after[p]['free'] - before[p]['free'] for p in after},
               retained=retained, archived_small_files=saved_files, archived_small_bytes=saved_bytes,
               local_large_files_created=0)
(AUDIT / 'receipt.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt, indent=2))
