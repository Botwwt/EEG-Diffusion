"""Preserve one inactive recomputable condition cache at a private RAM link."""
import json
import os
from pathlib import Path
import shutil
import stat
import time
from relocate_retired_ATM_features_CPU_20261021_v1 import ROOT, sha, memory, active_uses, write, GIB


def main():
    source = Path('/root/atm_diffusion_20261015/research/eeg_diffusion_next_20261018_v1/shared/conditions_EEG_ONLY.pt')
    items = json.loads((ROOT / 'audit/BrainHIVE_storage_recovery_kan_readonly_v1.json').read_text())['files']
    item = next(x for x in items if x['path'] == str(source))
    assert source.parent.resolve() == Path('/root/atm_diffusion_20261015/research/eeg_diffusion_next_20261018_v1/shared')
    before = source.lstat()
    assert stat.S_ISREG(before.st_mode) and before.st_nlink == 1
    assert before.st_size == item['bytes'] == 416810752 and before.st_mtime_ns == item['mtime_ns']
    assert sha(source, evict=True) == item['sha256']
    assert not active_uses([source])
    mem = memory()
    assert mem['maximum'] - mem['current'] - 2 * item['bytes'] >= 4 * GIB, mem
    assert mem['shm_free'] - item['bytes'] >= GIB, mem
    ram = Path('/dev/shm/brainhive_retired_shared_20261021_v1')
    receipt = ROOT / 'audit/BrainHIVE_retired_shared_RAM_relocation_20261021_v1.json'
    assert not ram.exists() and not receipt.exists()
    ram.mkdir(mode=0o700)
    target = ram / source.name
    record = dict(status='PREPARED_INACTIVE_DERIVED_CACHE_RELOCATION',
        UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        original=str(source), RAM_target=str(target), sha256=item['sha256'], bytes=item['bytes'],
        original_allocated_bytes=before.st_blocks * 512, original_manifest=item['manifest'],
        before_memory=mem, before_root_free=shutil.disk_usage(ROOT).free,
        data_kind='Derived EEG-only normalized inputs and full/fold mapped conditions; raw EEG and all learned encoder/map weights retained.',
        RAM_is_persistent_backup=False, regeneration_bit_identity_proven=False,
        restoration_source='/root/atm_diffusion_20261015/research/eeg_diffusion_next_20261018_v1/shared_conditions.py:215-262; export-only from preserved checkpoints, never rerun its training main.',
        original_EEG_learned_weights_or_historical_images_deleted=0)
    write(receipt, record)
    with source.open('rb') as inp, target.open('xb') as out:
        os.chmod(target, 0o600)
        shutil.copyfileobj(inp, out, 1 << 20)
        out.flush()
        os.fsync(out.fileno())
        os.posix_fadvise(inp.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    assert sha(target) == item['sha256']
    after = source.lstat()
    assert (after.st_ino, after.st_size, after.st_mtime_ns) == (before.st_ino, before.st_size, before.st_mtime_ns)
    assert not active_uses([source])
    link = source.with_name(source.name + '.RAM_link_pending_20261021_v1')
    assert not link.exists() and not link.is_symlink()
    os.symlink(str(target), link)
    os.replace(link, source)
    assert source.is_symlink() and sha(source) == item['sha256']
    from brainhive_lifecycle_storage_v1 import verify_remaining
    forecast = verify_remaining(json.loads((ROOT / 'configs/BrainHIVE_bounded_main_recipe_v1.json').read_text()))
    record.update(status='BYTE_IDENTICAL_LINK_AND_MAIN_FORECAST_VERIFIED', after_memory=memory(),
        after_root_free=shutil.disk_usage(ROOT).free, main_remaining_forecast=forecast)
    write(receipt, record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
