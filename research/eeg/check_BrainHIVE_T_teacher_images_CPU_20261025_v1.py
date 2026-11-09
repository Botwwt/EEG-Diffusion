"""Identity corruption and paired image-score join checks, without CUDA."""
import copy
import argparse
import csv
import json
from pathlib import Path
import torch
import diagnose_BrainHIVE_T_teacher_images_20261025_v1 as M

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--queries',type=Path,default=ROOT/'audit/diagnostic_memory_recovery_20261025_v3/completed/function/queries.json')
    args=parser.parse_args()
    rows=json.loads(args.queries.read_bytes())['rows']
    protocol=json.loads((ROOT/'configs/split_manifest.json').read_bytes())
    with (ROOT/'audit/image_manifest.csv').open(encoding='utf-8',newline='') as stream:
        metadata={int(r['array_row']):r for r in csv.DictReader(stream) if r['dataset_split']=='train'}
    M.verify_rows(rows,protocol,metadata)
    for field,value in [('negative_array_row',rows[0]['array_row']),('stimulus_id','corrupt.jpg'),('trajectory_seed',-1)]:
        bad=copy.deepcopy(rows);bad[0][field]=value
        try:M.verify_rows(bad,protocol,metadata)
        except AssertionError:pass
        else:raise AssertionError('Invalid identity accepted: '+field)
    ids=sorted({r['array_row'] for r in rows});positions={r:i for i,r in enumerate(ids)}
    target=torch.eye(8);features=[]
    for row in rows:features.extend([target[positions[row['array_row']]],target[positions[row['negative_array_row']]],torch.zeros(8)])
    scores=M.cosine_margin_rows(torch.stack(features),target,rows)
    assert len(scores)==240
    for record in scores:
        assert record['paired_cosine_margin']=={'correct_F':1.,'wrong_F':-1.,'missing':0.}[record['condition']]
    assert not torch.cuda.is_initialized()
    print('CPU_IDENTITY_CORRUPTION_AND_HAND_CONSTRUCTED_PAIRED_SCORE_CHECKS_PASSED_NO_CUDA')


if __name__=='__main__':main()
