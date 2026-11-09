"""Materialize exact audited target bytes once, separately from EEG inference."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parent
DATA=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--generation-manifest',required=True);parser.add_argument('--partition',choices=['V','benchmark'],required=True);args=parser.parse_args()
    if args.partition=='benchmark':assert (ROOT/'configs/FINAL_EVAL_UNLOCK.json').exists()
    source=[json.loads(line) for line in Path(args.generation_manifest).read_text().splitlines()]
    unique={}
    for row in source:
        key=row['stimulus_id']
        if key in unique:assert row['target_image_sha256']==unique[key]['target_image_sha256']
        else:unique[key]=row
    out=DATA/'evaluation/gallery'/args.partition;images=out/'images';images.mkdir(parents=True,exist_ok=True)
    records=[];archives={}
    for key,row in sorted(unique.items(),key=lambda item:int(item[1]['array_row'])):
        z=row['target_image_zip']
        if z not in archives:archives[z]=zipfile.ZipFile(z)
        raw=archives[z].read(row['target_image_member']);assert hashlib.sha256(raw).hexdigest()==row['target_image_sha256']
        image=images/f'{row["array_row"]:08d}.jpg'
        if image.exists():assert image.read_bytes()==raw
        else:image.write_bytes(raw)
        records.append(dict(stimulus_id=key,concept_id=row['concept_id'],array_row=row['array_row'],image_path=str(image),
            image_sha256=row['target_image_sha256'],source_zip=z,source_member=row['target_image_member']))
    for archive in archives.values():archive.close()
    expected=1660 if args.partition=='V' else 200;assert len(records)==expected
    data=''.join(json.dumps(row)+'\n' for row in records);manifest=out/'manifest.jsonl'
    if manifest.exists():assert manifest.read_text()==data
    else:manifest.write_text(data)
    print(json.dumps(dict(status='COMPLETED',gallery_images=expected,manifest=str(manifest))))


if __name__=='__main__':main()
