"""Losslessly archive a retired checkpoint, including every optimizer byte.

Simple reversible byte shuffle + standard zlib, not tensor quantization.
The original path keeps an exact RAM copy; the archive is durable.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import zlib

import numpy as np

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/root/autodl-tmp/avde_native680_stage2_20261016_v1/training/ar-ckpt-last.pth')
ARCHIVE=SOURCE.with_name('ar-ckpt-last.pth.EXACT_BS4_ZLIB_v1')
RAM=Path('/dev/shm/BrainHIVE_private_derived_cache_20261020_v2/lossless_preserved_checkpoints/ar-ckpt-last.pth')
MAGIC=b'EEG_BS4_ZLIB_V1\x00'
BLOCK=4<<20


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def shuffle(x,inverse=False):
    n=len(x)//4*4;a=np.frombuffer(x[:n],dtype=np.uint8)
    a=a.reshape(4,-1).T if inverse else a.reshape(-1,4).T
    return a.copy().tobytes()+x[n:]


def decompress(path,destination=None):
    h=hashlib.sha256();total=0
    with path.open('rb') as f:
        assert f.read(len(MAGIC))==MAGIC
        while True:
            header=f.read(8)
            if not header:break
            assert len(header)==8;n,c=struct.unpack('<II',header)
            payload=f.read(c);assert len(payload)==c
            raw=shuffle(zlib.decompress(payload),inverse=True);assert len(raw)==n
            h.update(raw);total+=n
            if destination is not None:destination.write(raw)
    return h.hexdigest(),total


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--restore-to-RAM',action='store_true');args=parser.parse_args()
    receipt=ROOT/'audit/BrainHIVE_old_AVDE_exact_checkpoint_archive_v13.json'
    if args.restore_to_RAM:
        d=json.loads(receipt.read_text());assert digest(ARCHIVE)==d['archive_sha256']
        RAM.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        if RAM.exists():assert digest(RAM)==d['original_sha256'];return
        assert shutil.disk_usage('/dev/shm').free>d['original_bytes']+(512<<20)
        temporary=RAM.with_name(RAM.name+'.restoring')
        with temporary.open('xb') as f:h,n=decompress(ARCHIVE,f)
        assert h==d['original_sha256'] and n==d['original_bytes'];temporary.replace(RAM)
        print('ORIGINAL_CHECKPOINT_BYTES_RESTORED_EXACTLY',flush=True);return
    assert SOURCE.is_file() and not SOURCE.is_symlink() and not receipt.exists() and not ARCHIVE.exists()
    original_stat=SOURCE.stat();before=shutil.disk_usage(SOURCE.parent).free
    assert shutil.disk_usage(SOURCE.parent).free>original_stat.st_size+(512<<20)
    assert shutil.disk_usage('/dev/shm').free>original_stat.st_size+(1<<30)
    h=hashlib.sha256();seen=0
    with SOURCE.open('rb') as source,ARCHIVE.open('xb') as target:
        target.write(MAGIC)
        while True:
            raw=source.read(BLOCK)
            if not raw:break
            h.update(raw);seen+=len(raw);payload=zlib.compress(shuffle(raw),level=1)
            target.write(struct.pack('<II',len(raw),len(payload)));target.write(payload)
            if seen%(256<<20)==0:print('LOSSLESS_CHECKPOINT_ARCHIVE_BYTES',seen,flush=True)
    original_sha=h.hexdigest();roundtrip,decoded_bytes=decompress(ARCHIVE)
    assert roundtrip==original_sha and decoded_bytes==seen==original_stat.st_size
    assert SOURCE.stat().st_mtime_ns==original_stat.st_mtime_ns
    assert ARCHIVE.stat().st_size<SOURCE.stat().st_size
    RAM.parent.mkdir(mode=0o700,parents=True,exist_ok=True);assert not RAM.exists()
    shutil.copyfile(SOURCE,RAM);os.chmod(RAM,0o600);assert digest(RAM)==original_sha
    record=dict(status='COMPLETE_CHECKPOINT_AND_OPTIMIZER_BYTE_EXACT_ARCHIVE_VERIFIED',original_path=str(SOURCE),
        original_bytes=seen,original_sha256=original_sha,archive_path=str(ARCHIVE),archive_bytes=ARCHIVE.stat().st_size,
        archive_sha256=digest(ARCHIVE),RAM_exact_alias=str(RAM),models_and_optimizer_all_preserved=True,
        restoration_entrypoint=str(Path(__file__))+' --restore-to-RAM',
        codec='4-byte-lane shuffle, independent zlib level1 blocks, little-endian original/compressed lengths; fully reversible',
        source_sha256=digest(__file__),raw_EEG_or_images_modified=False)
    receipt.write_text(json.dumps(record,indent=2)+'\n')
    link=SOURCE.with_name(SOURCE.name+'.EXACT_RAM_LINK_PENDING');assert not link.exists();link.symlink_to(RAM)
    os.replace(link,SOURCE);assert digest(SOURCE)==original_sha
    print(json.dumps(dict(status=record['status'],saved_bytes=seen-ARCHIVE.stat().st_size,
        persistent_free_before=before,persistent_free_after=shutil.disk_usage(SOURCE.parent).free)),flush=True)


if __name__=='__main__':main()
