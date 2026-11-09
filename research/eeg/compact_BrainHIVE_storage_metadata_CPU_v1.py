"""Lossless durable gzip of large completed cache manifests; exact RAM aliases."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
RAM=Path('/dev/shm/BrainHIVE_private_derived_cache_20261020_v2/storage_metadata')
NAMES=['BrainHIVE_derived_storage_rebalance_plan_v10.json','BrainHIVE_derived_storage_rebalance_plan_v11.json',
    'BrainHIVE_derived_storage_rebalance_v11.json','BrainHIVE_derived_storage_rebalance_progress_v11.jsonl']


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    receipt=ROOT/'audit/BrainHIVE_storage_metadata_exact_archive_v14.json';assert not receipt.exists()
    files=[ROOT/'audit'/n for n in NAMES];assert all(p.is_file() and not p.is_symlink() for p in files)
    assert shutil.disk_usage('/dev/shm').free>sum(p.stat().st_size for p in files)+(512<<20)
    RAM.mkdir(mode=0o700,exist_ok=False);records=[]
    for p in files:
        original=sha(p);archive=p.with_name(p.name+'.EXACT.gz');assert not archive.exists()
        with p.open('rb') as source,archive.open('xb') as target:
            with gzip.GzipFile(filename='',mode='wb',fileobj=target,compresslevel=1,mtime=0) as compressed:shutil.copyfileobj(source,compressed,4<<20)
        with gzip.open(archive,'rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==original
        raw=RAM/p.name;shutil.copyfile(p,raw);assert sha(raw)==original
        link=p.with_name(p.name+'.METADATA_RAM_PENDING');assert not link.exists();link.symlink_to(raw);os.replace(link,p)
        records.append(dict(path=str(p),sha256=original,bytes=raw.stat().st_size,archive=str(archive),archive_sha256=sha(archive),
            archive_bytes=archive.stat().st_size,RAM_exact_alias=str(raw),restore='gzip.open(archive) -> RAM_exact_alias, verify original SHA'))
    receipt.write_text(json.dumps(dict(status='ALL_COMPLETED_CACHE_MANIFEST_BYTES_PRESERVED_IN_DURABLE_GZIP',files=records,
        net_saved_bytes=sum(x['bytes']-x['archive_bytes'] for x in records),model_or_EEG_or_image_bytes_modified=False),indent=2)+'\n')
    print(json.dumps(dict(status='EXACT_MANIFEST_ARCHIVES_SAVED',net_saved_bytes=sum(x['bytes']-x['archive_bytes'] for x in records))),flush=True)


if __name__=='__main__':main()
