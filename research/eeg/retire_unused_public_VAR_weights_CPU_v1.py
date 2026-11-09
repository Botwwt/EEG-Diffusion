"""Retire only two verified, re-downloadable frozen VAR assets, never run results."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
OLD = Path('/root/rime_var_20261015').resolve()


def sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def main():
    manifest = OLD / 'assets/weight_manifest.json'
    items = json.loads(manifest.read_text())['assets']
    assert {a['path'] for a in items} == {'assets/var_d16.pth', 'assets/vae_ch160v4096z32.pth'}
    targets = [(OLD / a['path']).resolve() for a in items]
    assert all(p.parent == OLD/'assets' and not (OLD/a['path']).is_symlink() for a,p in zip(items, targets))
    references = []
    excluded = {os.getpid(), os.getppid()}
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit() or int(directory.name) in excluded:
            continue
        try:
            for descriptor in (directory/'fd').iterdir():
                try:
                    if descriptor.resolve() in targets:
                        references.append(str(descriptor))
                except OSError:
                    pass
            maps = (directory/'maps').read_text(errors='replace')
            if any(str(p) in maps for p in targets):
                references.append(str(directory/'maps'))
        except (OSError, PermissionError):
            pass
    assert not references, references
    verified = []
    for item, path in zip(items, targets):
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({'https': 'http://127.0.0.1:33343'}))
        request = urllib.request.Request(item['source'] + '?public_recovery_probe=20261020', headers={'Range': 'bytes=0-7'})
        with opener.open(request, timeout=25) as response:
            assert response.status == 206
            data = response.read(9)
        assert len(data) == 8 and data == path.open('rb').read(8)
        verified.append(item | dict(native_full_SHA_verified=True, pinned_public_recovery_range_verified=True))
    before = shutil.disk_usage('/root').free
    report = dict(status='VERIFIED_UNUSED_PUBLIC_FROZEN_VAR_WEIGHTS_READY_TO_RETIRE',
                  utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), manifest_sha256=sha(manifest),
                  assets=verified, open_file_or_mmap_references=references,
                  trained_checkpoints_deleted=False, EEG_or_image_data_deleted=False,
                  source_code_or_experiment_results_deleted=False, free_before=before)
    out = ROOT/'audit/unused_public_VAR_weights_retirement_20261020_v1.json'
    assert not out.exists()
    out.write_text(json.dumps(report, indent=2)+'\n')
    for path in targets:
        assert path.parent == OLD/'assets'
        path.unlink()
    after = shutil.disk_usage('/root').free
    report.update(status='RETIRED_TWO_VERIFIED_PUBLIC_FROZEN_VAR_WEIGHT_PAYLOADS',
                  free_after=after, observed_free_increase=after-before,
                  nominal_payload_bytes=sum(x['bytes'] for x in verified))
    out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','nominal_payload_bytes','observed_free_increase')}))


if __name__ == '__main__':
    main()
