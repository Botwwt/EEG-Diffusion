"""Guarded repository evidence delivery; no training, GPU, deletion or source edits."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import zipfile
R=Path(__file__).resolve().parent
P=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/rapid_iteration_20261027_v1/candidates')
def sha(p):
    with Path(p).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()
def ref(p): return dict(path=str(p),sha256=sha(p))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True); p.add_argument('--archive-sha256',required=True)
    p.add_argument('--science-spec',type=Path,required=True); p.add_argument('--receipt',type=Path,required=True)
    args=p.parse_args()
    assert args.archive.resolve().is_relative_to(P.resolve())
    assert args.science_spec.resolve().is_relative_to((R/'configs').resolve())
    assert args.receipt.resolve().is_relative_to((R/'audit').resolve()) and not args.receipt.exists()
    assert sha(args.archive)==args.archive_sha256
    spec_ref=ref(args.science_spec); spec=json.loads(args.science_spec.read_bytes())
    pins={Path(x['path']).resolve():x['sha256'] for x in spec['source_pins']}
    for path,digest in pins.items(): assert sha(path)==digest
    protected=pins|{args.science_spec.resolve():spec_ref['sha256']}
    with zipfile.ZipFile(args.archive) as archive:
        manifest=json.loads(archive.read('delivery_manifest.json')); entries=manifest['files']; assert entries
        data={}; paths=set()
        for item in entries:
            name=item['path']; path=R/name; resolved=path.resolve()
            assert resolved.is_relative_to(R.resolve()) and resolved not in paths; paths.add(resolved)
            raw=archive.read(name); assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
            if resolved in protected: assert item['sha256']==protected[resolved],'Delivery cannot change trained scientific sources'
            if path.exists() and sha(path)!=item['sha256']:
                assert item.get('expected_previous_sha256')==sha(path),'Unreviewed existing change: '+name
            assert not path.with_name(path.name+'.delivery_pending').exists()
            data[name]=raw
        changed=[]; backup_root=args.archive.parent/'document_delivery_before_v1'
        for item in entries:
            name=item['path']; path=R/name
            if path.exists() and sha(path)==item['sha256']: continue
            if path.exists():
                backup=backup_root/name; assert backup.resolve().is_relative_to(backup_root.resolve()) and not backup.exists()
                backup.parent.mkdir(parents=True,exist_ok=True); backup.write_bytes(path.read_bytes())
            path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+'.delivery_pending')
            with tmp.open('xb') as stream: stream.write(data[name]); stream.flush(); os.fsync(stream.fileno())
            os.replace(tmp,path); changed.append(name)
        for item in entries: assert sha(R/item['path'])==item['sha256']
        for path,digest in protected.items(): assert sha(path)==digest
    result=dict(status='REVIEWED_DELIVERY_ALL_FILE_SHAS_VERIFIED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source=ref(__file__),files=len(entries),changed=changed,after_sha256={x['path']:x['sha256'] for x in entries},
        archive=ref(args.archive),science_spec=spec_ref,science_source_count=len(pins),all_scientific_sources_unchanged=True,
        backup_root=str(backup_root),new_GPU_jobs=0,new_fit=0,formal_D_resumed=False)
    args.receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','files','science_source_count','all_scientific_sources_unchanged')}))


if __name__=='__main__': main()
