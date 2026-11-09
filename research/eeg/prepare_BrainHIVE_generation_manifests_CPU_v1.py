"""Separate allowed inference metadata from post-generation target joins."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
WORK=Path('/root/brainhive_reproduction_20261020_v1')


def main():
    protocol=json.loads((ROOT/'configs/split_manifest.json').read_text())
    images=list(csv.DictReader((ROOT/'audit/image_manifest.csv').open()))
    destination=WORK/'data/generation_inputs'
    destination.mkdir(parents=True,exist_ok=True)
    for split in ['benchmark','V','Tdiag']:
        if split=='benchmark':
            rows=sorted([r for r in images if r['dataset_split']=='test'],key=lambda r:int(r['array_row']))
        else:
            allowed=set(protocol['image_rows']['T' if split=='Tdiag' else 'V'])
            rows=[r for r in images if r['dataset_split']=='train' and int(r['array_row']) in allowed]
            rows=sorted(rows,key=(lambda r:hashlib.sha256(('1729|BrainHIVE_Tdiag|'+r['stimulus_id']).encode()).digest())
                        if split=='Tdiag' else lambda r:int(r['array_row']))
            if split=='Tdiag':rows=rows[:32]
        native=[dict(stimulus_id=r['stimulus_id'],array_row=int(r['array_row'])) for r in rows]
        payload=dict(split=split,inputs=native,expected_queries=32 if split=='Tdiag' else (2000 if split=='benchmark' else 16600),
                     repeat_protocol='R80' if split=='benchmark' else 'R4',
                     subjects=['sub-01'] if split=='Tdiag' else [f'sub-{i:02d}' for i in range(1,11)],
                     target_image_or_class_fields_present=False)
        out=destination/(split+'.json')
        text=json.dumps(payload,indent=2)+'\n'
        if out.exists():assert out.read_text()==text
        else:out.write_text(text)
        target=destination/(split+'_targets.jsonl')
        text=''.join(json.dumps(r)+'\n' for r in rows)
        if target.exists():assert target.read_text()==text
        else:target.write_text(text)
    print('INFERENCE_ONLY_AND_TARGET_MANIFESTS_SEPARATELY_PREPARED')


if __name__=='__main__':main()
