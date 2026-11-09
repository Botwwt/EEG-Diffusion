"""Losslessly compact one cold legacy input; preserve every original byte.

Uses the existing verified BS4/zlib archive codec. The durable archive is fully
decoded and SHA-verified before replacing the old file with an exact RAM alias.
--restore rehydrates that alias after volatile storage is cleared.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time
import zlib

from archive_retired_AVDE_checkpoint_losslessly_CPU_v1 import MAGIC, BLOCK, shuffle, decompress
from cleanup_BrainHIVE_baseline_preserving_CPU_20261022_v1 import sha, identity, live_uses

ROOT = Path(__file__).resolve().parent
SERVER = Path('/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1')
SOURCE = Path('/root/avde_author_training_cache_20261016/preprocessed_bc/sub-01/preprocessed_eeg_training.npy')
ARCHIVES = Path('/root/autodl-tmp/retained_legacy_exact_archives_20261022_v1')
ARCHIVE = ARCHIVES / 'AVDE_preprocessed_eeg_training.npy.EXACT_BS4_ZLIB_v1'
RAM_ROOT = Path('/dev/shm/BrainHIVE_private_derived_cache_20261020_v2/retained_legacy_inputs')
RAM = RAM_ROOT / 'AVDE_preprocessed_eeg_training.npy'
RECEIPT = ROOT / 'audit/old_AVDE_input_lossless_archive_20261022_v1.json'
EXPECTED_SHA = 'b7494ebe01fad73c61433b388388fb9cde7284a0eeb6055b519a24bad9b65fd4'
EXPECTED_BYTES = 3334466556


def save(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False); f.write('\n')
        f.flush(); os.fsync(f.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def restore():
    record = json.loads(RECEIPT.read_text())
    assert sha(ARCHIVE) == record['archive_sha256']
    RAM_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    if RAM.exists():
        assert sha(RAM) == record['original_sha256']
    else:
        assert shutil.disk_usage('/dev/shm').free > EXPECTED_BYTES + (512 << 20)
        temp = RAM.with_name(RAM.name + '.restoring')
        with temp.open('xb') as stream:
            digest, size = decompress(ARCHIVE, stream)
            stream.flush(); os.fsync(stream.fileno())
        assert (digest, size) == (record['original_sha256'], record['original_bytes'])
        os.chmod(temp, 0o600); temp.replace(RAM)
    if SOURCE.is_symlink():
        assert os.readlink(SOURCE) == str(RAM)
    else:
        assert not SOURCE.exists()
        SOURCE.symlink_to(RAM)
    assert sha(SOURCE) == EXPECTED_SHA
    print('LEGACY_INPUT_ORIGINAL_BYTES_RESTORED_FROM_DURABLE_ARCHIVE', flush=True)


def self_check():
    import tempfile
    for size in (0, 1, 2, 3, 4, 4097, 65539):
        raw = bytes((i * 37) % 256 for i in range(size))
        assert shuffle(shuffle(raw), inverse=True) == raw
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'test.archive'
            compressed = zlib.compress(shuffle(raw), level=1)
            p.write_bytes(MAGIC + struct.pack('<II', len(raw), len(compressed)) + compressed)
            out = io.BytesIO(); digest, decoded = decompress(p, out)
            assert out.getvalue() == raw and decoded == size
            assert digest == hashlib.sha256(raw).hexdigest()
    print('LOSSLESS_CODEC_ROUNDTRIP_ALL_TAIL_LENGTHS_PASSED')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--apply', action='store_true')
    parser.add_argument('--restore', action='store_true'); parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check: self_check(); return
    assert ROOT == SERVER
    if args.restore: restore(); return
    assert args.apply and not RECEIPT.exists() and not ARCHIVE.exists() and not RAM.exists()
    assert identity(SOURCE)[2:4] == (EXPECTED_BYTES, 1)
    before_identity = identity(SOURCE)
    assert sha(SOURCE) == EXPECTED_SHA
    process_cleanup = json.loads((ROOT / 'audit/legacy_process_files_cleanup_20261022_v1/receipt.json').read_text())
    assert process_cleanup['status'] == 'OLD_PROCESS_FILES_REMOVED_FINAL_MODELS_AND_RESULTS_PRESERVED'
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    assert not live_uses([str(SOURCE)])
    recipe = json.loads((ROOT / 'configs/functional_profiles_20261022_v1/O.json').read_text())
    assert str(SOURCE) not in recipe['input_sha256']
    assert shutil.disk_usage('/dev/shm').free > EXPECTED_BYTES * 2 + (1 << 30)
    RAM_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = RAM_ROOT / 'old_AVDE_input.archive.pending'
    assert not temporary.exists()
    original_hash = hashlib.sha256(); total = 0
    before = {p: shutil.disk_usage(p).free for p in ('/root', '/root/autodl-tmp')}
    with SOURCE.open('rb') as source, temporary.open('xb') as target:
        target.write(MAGIC)
        while True:
            raw = source.read(BLOCK)
            if not raw: break
            original_hash.update(raw); total += len(raw)
            compressed = zlib.compress(shuffle(raw), level=1)
            target.write(struct.pack('<II', len(raw), len(compressed))); target.write(compressed)
            if total % (256 << 20) == 0: print('LOSSLESS_BYTES_ENCODED', total, flush=True)
    assert (original_hash.hexdigest(), total) == (EXPECTED_SHA, EXPECTED_BYTES)
    assert temporary.stat().st_size + (64 << 20) < EXPECTED_BYTES
    assert decompress(temporary) == (EXPECTED_SHA, EXPECTED_BYTES)
    assert shutil.disk_usage('/root/autodl-tmp').free > temporary.stat().st_size + (256 << 20)
    ARCHIVES.mkdir(mode=0o700, exist_ok=True)
    assert ARCHIVES.resolve() == ARCHIVES
    with temporary.open('rb') as source, ARCHIVE.open('xb') as target:
        shutil.copyfileobj(source, target, length=4 << 20); target.flush(); os.fsync(target.fileno())
    os.chmod(ARCHIVE, 0o600)
    descriptor = os.open(ARCHIVES, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)
    assert sha(ARCHIVE) == sha(temporary)
    assert decompress(ARCHIVE) == (EXPECTED_SHA, EXPECTED_BYTES)
    # The sole original is untouched until the durable roundtrip and exact alias pass.
    with RAM.open('xb') as target:
        digest, decoded = decompress(ARCHIVE, target)
        target.flush(); os.fsync(target.fileno())
    os.chmod(RAM, 0o600)
    assert (digest, decoded) == (EXPECTED_SHA, EXPECTED_BYTES) and sha(RAM) == EXPECTED_SHA
    assert identity(SOURCE) == before_identity and not live_uses([str(SOURCE)])
    record = dict(status='DURABLE_ARCHIVE_AND_ORIGINAL_BYTE_SHA_VERIFIED_BEFORE_PATH_SWAP',
        UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), source_sha256=sha(__file__),
        original_path=str(SOURCE), original_sha256=EXPECTED_SHA, original_bytes=EXPECTED_BYTES,
        archive_path=str(ARCHIVE), archive_sha256=sha(ARCHIVE), archive_bytes=ARCHIVE.stat().st_size,
        RAM_exact_alias=str(RAM), codec='Existing blockwise four-byte-lane shuffle plus zlib1; no tensor quantization',
        restore_argv=['/root/miniconda3/bin/python', str(Path(__file__).resolve()), '--restore'],
        historical_results_images_models_and_EEG_values_preserved=True, before_free=before)
    save(RECEIPT, record)
    pending = SOURCE.with_name(SOURCE.name + '.EXACT_RAM_LINK_PENDING_20261022_v1')
    assert not pending.exists() and not pending.is_symlink()
    pending.symlink_to(RAM); os.replace(pending, SOURCE)
    assert sha(SOURCE) == EXPECTED_SHA and sha(ARCHIVE) == record['archive_sha256']
    temporary.unlink()
    after = {p: shutil.disk_usage(p).free for p in before}
    complete = dict(status='OLD_AVDE_INPUT_LOSSLESS_ARCHIVE_COMPLETE_ORIGINAL_BYTES_RETAINED',
        UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), archive_receipt_sha256=sha(RECEIPT),
        after_free=after, actual_aggregate_free_delta=sum(after.values()) - sum(before.values()),
        source_path_is_exact_RAM_alias=True, durable_archive_independently_decoded_and_SHA_verified=True,
        no_new_GPU_work=True, formal_training_started=False)
    save(RECEIPT.with_name('old_AVDE_input_lossless_archive_complete_20261022_v1.json'), complete)
    print(json.dumps(complete), flush=True)


if __name__ == '__main__': main()
