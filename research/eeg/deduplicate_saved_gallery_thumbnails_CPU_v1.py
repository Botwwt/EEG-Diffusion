"""Reclaim duplicate thumbnail storage without changing any historical bytes.

Only two named, completed historical galleries are eligible. All original image
files, pages, scores, labels, and paths remain. Current cache/new-gallery inodes
are never replaced, preserving the already verified thumbnail-reuse contract.
"""
from pathlib import Path
import csv
import hashlib
import json
import os
import shutil
import time
import uuid

ROOT = Path(__file__).resolve().parent
BASE = Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/galleries')
CANONICAL = ['IDEA1_Conditional_RealPaired_V_complete_20261019T231408Z_v1',
             'IDEA1_Frozen_RealPaired_Conditional_complete_20261020_v2']
TARGETS = ['IDEA1_Conditional_V_complete_20261019T165807Z_v1',
           'V_existing_REG_patience20_complete_v1']


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def thumbs(name):
    directory = BASE/name
    assert directory.resolve() == directory and not directory.is_symlink()
    proof = json.loads((directory/'completion.json').read_text())
    assert proof['status'] == 'COMPLETED_FULL_OFFLINE_GALLERY'
    assert proof['queries'] == 16600 and proof['missing_images'] == proof['corrupt_images'] == 0
    return sorted((directory/'blinded/thumbs').glob('*.jpg'))


def main():
    start = time.monotonic()
    receipt = ROOT/'audit/gallery_thumbnail_deduplication_20261020_v1.json'
    inventory = ROOT/'audit/gallery_thumbnail_deduplication_paths_20261020_v1.csv'
    assert not receipt.exists() and not inventory.exists()
    assert BASE.resolve() == BASE
    free = shutil.disk_usage(BASE).free
    index, hashed = {}, {}
    for name in CANONICAL:
        for p in thumbs(name):
            assert not p.is_symlink() and p.resolve().is_relative_to(BASE/name/'blinded/thumbs')
            stat = p.stat()
            inode = (stat.st_dev, stat.st_ino)
            digest = hashed.get(inode)
            if digest is None:
                digest = sha(p)
                hashed[inode] = digest
            index.setdefault((stat.st_size, digest), p)
    actions = examined = unchanged = released = 0
    with inventory.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['historical_path','canonical_path','sha256','bytes',
                                             'before_inode','after_inode','action'])
        writer.writeheader()
        for name in TARGETS:
            for p in thumbs(name):
                assert not p.is_symlink() and p.resolve().parent == BASE/name/'blinded/thumbs'
                stat = p.stat()
                digest = sha(p)
                source = index.get((stat.st_size, digest))
                examined += 1
                record = dict(historical_path=str(p), canonical_path=str(source) if source else '',
                              sha256=digest, bytes=stat.st_size, before_inode=stat.st_ino,
                              after_inode=stat.st_ino, action='UNCHANGED')
                if source is not None and stat.st_nlink == 1:
                    canonical = source.stat()
                    assert stat.st_dev == canonical.st_dev
                    assert sha(source) == digest
                    assert p.stat().st_ino == stat.st_ino and p.stat().st_mtime_ns == stat.st_mtime_ns
                    tmp = p.parent/('.dedup-'+uuid.uuid4().hex)
                    assert not tmp.exists()
                    try:
                        os.link(source, tmp)
                        assert sha(tmp) == digest and tmp.stat().st_size == stat.st_size
                        os.replace(tmp, p)
                    finally:
                        if tmp.exists():
                            tmp.unlink()
                    assert sha(p) == digest and p.stat().st_ino == canonical.st_ino
                    record.update(action='EXACT_BYTES_PRESERVED_HARDLINKED_DUPLICATE', after_inode=p.stat().st_ino)
                    actions += 1
                    released += stat.st_blocks*512
                else:
                    unchanged += 1
                writer.writerow(record)
    result = dict(status='COMPLETED_EXACT_BYTE_PRESERVING_HISTORICAL_THUMBNAIL_DEDUPLICATION',
                  canonical_galleries=CANONICAL, historical_galleries=TARGETS,
                  examined=examined, replaced_duplicate_storage=actions, unchanged=unchanged,
                  exact_duplicate_file_allocated_bytes_released=released,
                  free_bytes_before=free, free_bytes_after=shutil.disk_usage(BASE).free,
                  all_historical_thumbnail_paths_and_SHA256_preserved=True,
                  original_images_deleted=0, source_or_checkpoint_files_deleted=0,
                  GPU_used=False, source_sha256=sha(__file__), inventory=str(inventory),
                  inventory_sha256=sha(inventory), seconds=time.monotonic()-start)
    receipt.write_text(json.dumps(result, indent=2)+'\n')
    with (ROOT/'audit/CPU_postprocessing_ledger.jsonl').open('a') as f:
        f.write(json.dumps(dict(stage='exact_byte_preserving_gallery_thumbnail_deduplication',
                               GPU_used=False, seconds=result['seconds'], receipt=str(receipt)))+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
