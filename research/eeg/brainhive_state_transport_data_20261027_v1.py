"""D18 reuse complete native OOF inputs and original full-parent Dfit conditions."""
import csv
from pathlib import Path
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U
R=Path(__file__).resolve().parent

def training_inputs(device):
    from brainhive_component_data_v1 import load_training
    result=load_training(device);assert result['condition'].shape==(132300,1024) and result['target'].shape==(13230,1024)
    metadata={int(x['array_row']):x for x in csv.DictReader((R/'audit/image_manifest.csv').open()) if x['dataset_split']=='train'}
    return result,metadata

def development_inputs(spec,device):
    previous=U.read(U.checked(spec['D17_fit']));mapping=U.read(U.checked(previous['Dfit_queries']));queries=mapping['queries']
    contract=U.read(U.checked(spec['native_input_contract']));conditions=[]
    for si,subject in enumerate(contract['subjects']):
        assert subject['subject']==f'sub-{si+1:02d}'
        source=next(x for x in subject['sources'] if x['role']=='full')
        cache=torch.load(U.checked(dict(path=source['condition_cache_path'],sha256=source['condition_cache_sha256'])),weights_only=True,map_location='cpu',mmap=True)
        U.checked(dict(path=source['checkpoint_path'],sha256=source['checkpoint_sha256']))
        rows=source['groups']['calibration_only_Dfit'];ids=[x['array_row'] for x in rows];idx=[x['cache_row'] for x in rows]
        assert cache['array_rows'][idx].tolist()==ids==[q['array_row'] for q in queries[si*820:(si+1)*820]]
        conditions.append(cache['condition'][idx].clone())
    raw=torch.cat(conditions).to(device);target=torch.from_numpy(np.load(U.checked(previous['targets']),allow_pickle=False)).to(device)
    assert raw.shape==target.shape==(8200,1024)
    for start in range(0,8200,10):
        group=queries[start:start+10]
        assert len({(q['subject_id'],q['concept_id']) for q in group})==1
        assert len({q['array_row'] for q in group})==10
    return dict(raw=raw,target=target,queries=queries,groups=mapping['groups'],previous_target_source=previous['targets'])
